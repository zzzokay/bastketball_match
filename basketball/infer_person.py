#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN 人体检测推理脚本 — 最小可用版本

功能：
    从 USB 摄像头（左/右）读取画面，
    使用 RKNN 模型检测画面中的人（person），
    在 LCD 上实时显示检测结果。

    python3 infer_person.py \
     --device /dev/video43 \
     --model /home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn \
     --conf 0.25 \
    --nms 0.45

用法：
    # 使用左摄像头（USB2）
    python3 infer_person.py --device /dev/video43

    # 使用右摄像头（USB）
    python3 infer_person.py --device /dev/video0

    # 全屏显示
    python3 infer_person.py --device /dev/video43 --fullscreen

    # 调整检测阈值
    python3 infer_person.py --device /dev/video43 --conf 0.5

按键：
    q / ESC：退出
"""

import sys
import time
import argparse
import os
import signal
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# =============================================================================
# 退出信号处理：Ctrl+C 第一次请求退出，第二次强制退出
# =============================================================================
STOP_REQUESTED = False
SIGINT_COUNT = 0

def handle_exit_signal(signum, frame):
    global STOP_REQUESTED, SIGINT_COUNT
    SIGINT_COUNT += 1
    STOP_REQUESTED = True

    if SIGINT_COUNT >= 2:
        print("\n[信息] 再次收到 Ctrl+C，强制退出")
        os._exit(130)

    print("\n[信息] 收到 Ctrl+C，当前帧结束后退出；如果卡住请再按一次 Ctrl+C")



# =============================================================================
# 1. 配置常量
# =============================================================================

# 默认模型路径和标签路径
DEFAULT_MODEL = "/home/elf/work/basketball/model/basketball_player_2.1.0.rknn"
DEFAULT_LABELS = "/home/elf/work/basketball/model/labels.txt"

# 模型输入尺寸（YOLOv8 默认 640x640）
MODEL_INPUT_SIZE = 640

# COCO 数据集中 person 类的类别 ID（从 0 开始）
PERSON_CLASS_ID = 0


# =============================================================================
# 2. 工具函数
# =============================================================================

def sigmoid(x):
    """
    Sigmoid 激活函数：将任意实数映射到 (0, 1) 区间。

    YOLO 模型输出的类别分数通常是 raw logits，
    需要经过 sigmoid 才能变成概率值。
    """
    return 1.0 / (1.0 + np.exp(-x))


def load_labels(labels_path):
    """
    从文本文件加载标签。

    每行一个类别名称，例如：
        person
        bicycle
        car
        ...

    返回：标签列表，例如 ["person", "bicycle", ...]
    """
    path = Path(labels_path)
    if not path.exists():
        print(f"[警告] 标签文件不存在: {labels_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    return labels


def letterbox(image, target_size):
    """
    等比缩放 + 填充（Letterbox），将图像调整为模型输入尺寸。

    这是 YOLO 系列模型标准的预处理方式：
    1. 按原始宽高比等比缩放，使图像能放入 target_size x target_size 的画布
    2. 用灰色（114）填充剩余区域
    3. 返回填充后的图像和缩放参数（用于后处理时还原坐标）

    参数：
        image: 原始 BGR 图像
        target_size: 目标尺寸，例如 640

    返回：
        canvas: 填充后的正方形图像 (target_size, target_size, 3)
        ratio: 缩放比例
        pad_w: 水平方向填充像素数
        pad_h: 垂直方向填充像素数
    """
    h, w = image.shape[:2]

    # 计算缩放比例，取宽高中较小的那个，确保图像能放入目标尺寸
    ratio = min(target_size / w, target_size / h)

    # 计算缩放后的实际尺寸
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    # 等比缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建灰色画布（114 是 YOLO 的默认填充色）
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # 计算填充偏移，将缩放后的图像居中放置
    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0
    left = int(round(pad_w - 0.1))
    top = int(round(pad_h - 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas, ratio, pad_w, pad_h


def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制（Non-Maximum Suppression）。

    当多个检测框重叠度很高时，只保留分数最高的那个，
    去除重复检测。

    参数：
        boxes: 检测框数组，shape (N, 4)，格式 [x1, y1, x2, y2]
        scores: 置信度数组，shape (N,)
        iou_threshold: IoU 阈值，超过此值的框会被抑制

    返回：
        keep: 保留的检测框索引列表
    """
    if len(boxes) == 0:
        return []

    # 计算每个框的面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 按分数从高到低排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 取分数最高的框
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 计算当前框与其余框的 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # IoU = 交集 / 并集
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        # 保留 IoU 低于阈值的框（即不重复的框）
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# =============================================================================
# 3. RKNN 推理器
# =============================================================================

class PersonDetector:
    """
    基于 RKNN 的人体检测器。

    工作流程：
        1. 加载 RKNN 模型
        2. 对输入图像做 letterbox 预处理
        3. 送入 NPU 推理
        4. 解析 YOLO 输出，提取 person 类检测框
        5. NMS 去重
        6. 坐标映射回原图尺寸
    """

    def __init__(self, model_path, labels_path, obj_thresh, nms_thresh, use_rgb=True):
        """
        初始化检测器。

        参数：
            model_path: RKNN 模型文件路径
            labels_path: 标签文件路径
            obj_thresh: 目标置信度阈值（低于此值的检测结果会被过滤）
            nms_thresh: NMS IoU 阈值（重叠度高于此值的框会被合并）
            use_rgb: 是否将输入从 BGR 转为 RGB（RKNN 通常需要 RGB 输入）
        """
        self.labels = load_labels(labels_path)
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.use_rgb = use_rgb
        self._printed_output_shape = False

        # ---- 加载 RKNN 模型 ----
        self.rknn = RKNNLite()

        print(f"[信息] 正在加载模型: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载模型失败: ret={ret}")

        # 初始化运行时环境，使用全部 NPU 核心（RK3588 有 3 个 NPU 核心）
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            print("[警告] 多核初始化失败，退回单核模式")
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"初始化运行时失败: ret={ret}")

        print(f"[信息] 模型加载成功，输入尺寸: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
        print(f"[信息] 标签: {self.labels}")

    def close(self):
        """释放 RKNN 资源。"""
        try:
            self.rknn.release()
        except Exception:
            pass

    def detect(self, image_bgr):
        """
        对单帧图像进行人体检测。

        参数：
            image_bgr: BGR 格式的原始图像

        返回：
            results: 检测结果列表，每个元素为 (class_id, score, (x1, y1, x2, y2))
                     坐标已经映射回原图尺寸
        """
        orig_h, orig_w = image_bgr.shape[:2]

        # ---- Step 1: Letterbox 预处理 ----
        # 将原始图像等比缩放 + 填充为模型输入尺寸
        canvas, ratio, pad_w, pad_h = letterbox(image_bgr, MODEL_INPUT_SIZE)

        # BGR -> RGB（RKNN 模型通常期望 RGB 输入）
        if self.use_rgb:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # 添加 batch 维度：(H, W, C) -> (1, H, W, C)
        blob = np.expand_dims(canvas, axis=0)
        blob = np.ascontiguousarray(blob)

        # ---- Step 2: RKNN 推理 ----
        try:
            outputs = self.rknn.inference(inputs=[blob], data_format=['nhwc'])
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[错误] RKNN 推理失败: {e}")
            return []
        if outputs is None or len(outputs) == 0:
            return []

        # ---- Step 3: 解析 YOLO11 输出 ----
        output = outputs[0]

        if not self._printed_output_shape:
            print(f"[调试] RKNN 输出 shape: {output.shape}, dtype: {output.dtype}")
            self._printed_output_shape = True

        pred = np.squeeze(output)

        if pred.ndim != 2:
            print(f"[警告] 暂不支持的输出维度: {output.shape}")
            return []

        # NCHW/CHW: (84, 8400) -> 转置成 (8400, 84)
        # NHWC/HWC: (8400, 84) -> 直接用
        if pred.shape[0] < pred.shape[1] and pred.shape[0] >= 5:
            output_2d = pred.T
        elif pred.shape[1] >= 5:
            output_2d = pred
        else:
            print(f"[警告] 无法解析的 YOLO 输出 shape: {output.shape}")
            return []

        num_classes = output_2d.shape[1] - 4
        if num_classes <= 0:
            print(f"[警告] 类别数异常: {num_classes}, output_2d.shape={output_2d.shape}")
            return []

        boxes_xywh = output_2d[:, :4].astype(np.float32)
        cls_scores = output_2d[:, 4:].astype(np.float32)

        if not hasattr(self, "_debug_score_printed"):
            self._debug_score_printed = False

        if not self._debug_score_printed:
            print("[调试] output shape:", output.shape)
            print("[调试] output_2d shape:", output_2d.shape)
            print("[调试] cls_scores min/max/mean:",
                float(cls_scores.min()),
                float(cls_scores.max()),
                float(cls_scores.mean()))

            if cls_scores.ndim == 2 and cls_scores.shape[1] > 0:
                per_class_max = cls_scores.max(axis=0)
                top_ids = np.argsort(per_class_max)[::-1][:5]
                print("[调试] top class max:")
                for cid in top_ids:
                    print("  class", int(cid), "max_score", float(per_class_max[cid]))

                for t in [0.01, 0.05, 0.1, 0.25, 0.45]:
                    print(f"[调试] person score > {t}:",
                        int((cls_scores[:, PERSON_CLASS_ID] > t).sum()))

            self._debug_score_printed = True

        # YOLOv8/YOLO11 导出的输出通常已经是 0~1 概率
        # 只有明显超出范围才做 sigmoid
        if cls_scores.size == 0:
            return []

        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)
        # ---- Step 4: 提取 person 类（class_id=0）的检测结果 ----
        # 只保留 person 类的分数
        person_scores = cls_scores[:, PERSON_CLASS_ID]

        # 过滤低置信度检测框
        mask = person_scores > self.obj_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        person_scores = person_scores[mask]

        # 将 xywh 格式转换为 x1y1x2y2 格式
        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # ---- Step 5: NMS 去重 ----
        keep = nms(boxes, person_scores, self.nms_thresh)

        # ---- Step 6: 坐标映射回原图 ----
        # 去掉 letterbox 的填充偏移
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h

        # 除以缩放比例，还原到原图坐标
        boxes[:, :4] /= ratio

        # 裁剪到原图范围内
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        # 组装返回结果
        results = []
        for k in keep:
            x1, y1, x2, y2 = boxes[k]
            score = float(person_scores[k])
            results.append((PERSON_CLASS_ID, score, (int(x1), int(y1), int(x2), int(y2))))

        # 按置信度从高到低排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results


# =============================================================================
# 4. 摄像头打开函数
# =============================================================================

def open_camera(device, width=1920, height=1080, fps=30):
    """
    打开 USB 摄像头。

    使用 V4L2 后端打开摄像头，并设置分辨率和帧率。

    参数：
        device: 设备节点路径，例如 "/dev/video43"
        width: 期望宽度
        height: 期望高度
        fps: 期望帧率

    返回：
        cv2.VideoCapture 对象
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device}")

    # 设置 MJPEG 编码（USB 摄像头通常支持，传输带宽更小）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 读取实际设置的参数（摄像头可能不支持你请求的分辨率）
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[信息] 摄像头已打开: {device}")
    print(f"[信息] 实际分辨率: {real_w}x{real_h}, 帧率: {real_fps:.1f}fps")

    return cap


# =============================================================================
# 5. FPS 计数器
# =============================================================================

class FPSCounter:
    """简单的 FPS 计数器，使用指数平滑。"""

    def __init__(self):
        self.last_time = None
        self.fps = 0.0

    def update(self):
        """每调用一次更新一次 FPS，返回当前 FPS。"""
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                inst = 1.0 / dt
                # 指数平滑：0.9 * 旧值 + 0.1 * 新值，避免抖动
                self.fps = inst if self.fps == 0.0 else (0.9 * self.fps + 0.1 * inst)
        self.last_time = now
        return self.fps


# =============================================================================
# 6. 主函数
# =============================================================================

def main():
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    # ---- 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="RKNN 人体检测 — USB 摄像头实时推理")

    parser.add_argument(
        "--device", type=str, required=True,
        help="摄像头设备节点，例如 /dev/video43（左USB2）或 /dev/video0（右USB）"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="RKNN 模型文件路径"
    )
    parser.add_argument(
        "--labels", type=str, default=DEFAULT_LABELS,
        help="标签文件路径"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="置信度阈值（默认 0.25）"
    )
    parser.add_argument(
        "--nms", type=float, default=0.45,
        help="NMS IoU 阈值（默认 0.45）"
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="摄像头采集宽度（默认 1920）"
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="摄像头采集高度（默认 1080）"
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="全屏显示"
    )
    parser.add_argument(
        "--bgr-input", action="store_true",
        help="模型输入为 BGR 格式（默认 RGB）"
    )

    args = parser.parse_args()

    # ---- 初始化检测器 ----
    detector = PersonDetector(
        model_path=args.model,
        labels_path=args.labels,
        obj_thresh=args.conf,
        nms_thresh=args.nms,
        use_rgb=not args.bgr_input,
    )

    # ---- 打开摄像头 ----
    cap = open_camera(args.device, args.width, args.height)

    # ---- 初始化显示窗口 ----
    window_name = "Person Detection - RKNN"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps_counter = FPSCounter()

    print("[信息] 开始检测，按 q 或 ESC 退出")

    # ---- 主循环：采集 -> 推理 -> 显示 ----
    try:
        while not STOP_REQUESTED:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("[警告] 读取帧失败，重试...")
                continue

            # 推理
            results = detector.detect(frame)

            if STOP_REQUESTED:
                break

            # 在画面上绘制检测框
            vis = frame.copy()
            for class_id, score, (x1, y1, x2, y2) in results:
                # 画绿色矩形框
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签背景
                label = f"person {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), (0, 255, 0), -1)

                # 绘制标签文字
                cv2.putText(vis, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # 显示 FPS 和检测数量
            fps = fps_counter.update()
            info_text = f"FPS: {fps:.1f}  Persons: {len(results)}"
            cv2.putText(vis, info_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 显示画面
            cv2.imshow(window_name, vis)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print("[信息] 用户退出")
                break

    except KeyboardInterrupt:
        print("\n[信息] Ctrl+C 退出")

    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[信息] 程序已退出")


if __name__ == "__main__":
    main()
