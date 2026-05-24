#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dual_rknn_raw_detect.py

作用：
    只负责“双路原始视角 RKNN 人体检测”。

    这个脚本不做双目矫正、不做拼接、不做宽幅坐标映射。
    它的输出坐标仍然是左右摄像头原始画面中的 raw 坐标。

为什么单独拆出来：
    后续你从测试模型切换到正式 RKNN 模型时，只需要改这里的 RKNN 推理部分；
    坐标映射、宽图画框、运镜逻辑可以保持不变。

典型运行：
    cd /home/elf/work/basketball/camera_movement

    python3 dual_rknn_raw_detect.py \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --model /home/elf/work/basketball/model/basketball_player_2.1.0.rknn \
        --labels /home/elf/work/basketball/model/labels.txt \
        --width 1920 \
        --height 1080 \
        --fps 30 \
        --conf 0.25 \
        --nms 0.45 \
        --display

输出：
    1. 屏幕显示左右原始画面上的检测框。
    2. 可选输出 JSONL，每行一帧，里面是 left/right raw 检测框。

注意：
    这个脚本中的 PersonDetector 来自你已经跑通的 infer_person.py，
    只是在结构上整理成可复用模块，并增加了双路摄像头支持。
"""

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# RK3588 / Mali 平台上，OpenCV OpenCL 偶尔会引入额外开销或报错。
# 这里保持和你之前实时拼接脚本一致，关闭 OpenCL。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# =============================================================================
# 1. 全局参数与退出控制
# =============================================================================

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"
DEFAULT_MODEL = "/home/elf/work/basketball/model/basketball_player_2.1.0.rknn"
DEFAULT_LABELS = "/home/elf/work/basketball/model/labels.txt"

MODEL_INPUT_SIZE = 640
PERSON_CLASS_ID = 0

STOP_REQUESTED = False
SIGINT_COUNT = 0


def handle_exit_signal(signum, frame):
    """Ctrl+C 退出处理：第一次请求退出，第二次强制退出。"""
    global STOP_REQUESTED, SIGINT_COUNT
    SIGINT_COUNT += 1
    STOP_REQUESTED = True

    if SIGINT_COUNT >= 2:
        print("\n[信息] 再次收到 Ctrl+C，强制退出")
        os._exit(130)

    print("\n[信息] 收到退出信号，当前帧结束后退出；如果卡住请再按一次 Ctrl+C")


# =============================================================================
# 2. YOLO/RKNN 后处理工具函数
# =============================================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid 激活函数。

    有些 RKNN 导出的 YOLO 输出已经是 0~1 概率，
    有些输出是 raw logits。脚本后面会根据范围判断是否需要 sigmoid。
    """
    return 1.0 / (1.0 + np.exp(-x))


def load_labels(labels_path: str) -> List[str]:
    """加载标签文件，每行一个类别名称。"""
    path = Path(labels_path)
    if not path.exists():
        print(f"[警告] 标签文件不存在: {labels_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def letterbox(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, float, float]:
    """
    YOLO 标准预处理：等比缩放 + 灰色填充。

    参数：
        image:
            原始 BGR 图像。
        target_size:
            模型输入尺寸，例如 640。

    返回：
        canvas:
            target_size x target_size 的 BGR 图像。
        ratio:
            原图到模型输入的缩放比例。
        pad_w / pad_h:
            letterbox 左右/上下填充量，用于把检测框还原回原图。
    """
    h, w = image.shape[:2]
    ratio = min(target_size / w, target_size / h)

    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0

    left = int(round(pad_w - 0.1))
    top = int(round(pad_h - 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas, ratio, pad_w, pad_h


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    非极大值抑制。

    boxes 格式：[x1, y1, x2, y2]
    返回需要保留的索引。
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        remain = np.where(iou <= iou_threshold)[0]
        order = order[remain + 1]

    return keep


def parse_core_mask(core: str):
    """
    把命令行中的 NPU 核心字符串转换为 RKNNLite core_mask。

    可选值：
        all / 0_1_2 : 使用全部 3 个 NPU 核心
        0           : 只使用 NPU core 0
        1           : 只使用 NPU core 1
        2           : 只使用 NPU core 2
        auto        : 不传 core_mask，由 RKNNLite 自己决定

    双路检测时，如果两个检测器都用 all，可能互相抢 NPU。
    推荐先用：left-core=0，right-core=1。
    """
    core = str(core).strip().lower()
    if core in ("all", "0_1_2", "012"):
        return RKNNLite.NPU_CORE_0_1_2
    if core == "0":
        return RKNNLite.NPU_CORE_0
    if core == "1":
        return RKNNLite.NPU_CORE_1
    if core == "2":
        return RKNNLite.NPU_CORE_2
    if core == "auto":
        return None
    raise ValueError(f"不支持的 NPU core 参数: {core}")


# =============================================================================
# 3. RKNN 人体检测器
# =============================================================================

class PersonDetector:
    """
    基于 RKNNLite 的 person 检测器。

    这个类的 detect() 返回 raw 原始图像坐标：
        (class_id, score, (x1, y1, x2, y2))

    坐标还没有经过双目矫正，也没有映射到宽幅图。
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        obj_thresh: float,
        nms_thresh: float,
        use_rgb: bool = True,
        core: str = "auto",
        name: str = "detector",
    ):
        self.name = name
        self.labels = load_labels(labels_path)
        self.obj_thresh = float(obj_thresh)
        self.nms_thresh = float(nms_thresh)
        self.use_rgb = bool(use_rgb)
        self._printed_output_shape = False
        self._debug_score_printed = False

        self.rknn = RKNNLite()

        print(f"[{self.name}] 加载 RKNN 模型: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"[{self.name}] 加载 RKNN 模型失败: ret={ret}")

        core_mask = parse_core_mask(core)
        if core_mask is None:
            print(f"[{self.name}] 初始化 RKNN runtime: auto")
            ret = self.rknn.init_runtime()
        else:
            print(f"[{self.name}] 初始化 RKNN runtime: core={core}")
            ret = self.rknn.init_runtime(core_mask=core_mask)

        if ret != 0:
            print(f"[{self.name}] 指定 core 初始化失败，尝试 auto runtime")
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"[{self.name}] 初始化 RKNN runtime 失败: ret={ret}")

        print(f"[{self.name}] 模型加载成功，输入尺寸: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
        print(f"[{self.name}] 标签数量: {len(self.labels)}")

    def close(self) -> None:
        """释放 RKNN 资源。"""
        try:
            self.rknn.release()
        except Exception:
            pass

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        对一帧 BGR 图像进行 person 检测。

        返回：
            List[(class_id, score, (x1, y1, x2, y2))]
            其中坐标是输入图像的 raw 坐标。
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        orig_h, orig_w = image_bgr.shape[:2]

        # 1. 预处理：letterbox 到 640x640。
        canvas, ratio, pad_w, pad_h = letterbox(image_bgr, MODEL_INPUT_SIZE)

        # 2. BGR -> RGB。
        #    你的 infer_person.py 默认 use_rgb=True，这里保持一致。
        if self.use_rgb:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        blob = np.expand_dims(canvas, axis=0)
        blob = np.ascontiguousarray(blob)

        # 3. RKNN 推理。
        try:
            outputs = self.rknn.inference(inputs=[blob], data_format=["nhwc"])
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[{self.name}] RKNN 推理失败: {e}")
            return []

        if outputs is None or len(outputs) == 0:
            return []

        output = outputs[0]
        if not self._printed_output_shape:
            print(f"[{self.name}] RKNN 输出 shape: {output.shape}, dtype: {output.dtype}")
            self._printed_output_shape = True

        pred = np.squeeze(output)
        if pred.ndim != 2:
            print(f"[{self.name}] 暂不支持的输出维度: {output.shape}")
            return []

        # YOLOv8/YOLO11 常见输出：
        #   (84, 8400) 需要转置成 (8400, 84)
        #   (8400, 84) 可以直接用
        if pred.shape[0] < pred.shape[1] and pred.shape[0] >= 5:
            output_2d = pred.T
        elif pred.shape[1] >= 5:
            output_2d = pred
        else:
            print(f"[{self.name}] 无法解析 YOLO 输出 shape: {output.shape}")
            return []

        num_classes = output_2d.shape[1] - 4
        if num_classes <= PERSON_CLASS_ID:
            print(f"[{self.name}] 类别数异常: {num_classes}")
            return []

        boxes_xywh = output_2d[:, :4].astype(np.float32)
        cls_scores = output_2d[:, 4:].astype(np.float32)

        # 4. 如果输出明显不是概率，则做 sigmoid。
        if cls_scores.size == 0:
            return []
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)

        person_scores = cls_scores[:, PERSON_CLASS_ID]
        mask = person_scores > self.obj_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        person_scores = person_scores[mask]

        # xywh -> x1y1x2y2，坐标还在 letterbox 画布坐标系。
        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        boxes = np.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            axis=1,
        )

        # 5. NMS。
        keep = nms(boxes, person_scores, self.nms_thresh)
        if len(keep) == 0:
            return []

        # 6. 坐标还原回输入图像 raw 坐标。
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes[:, :4] /= ratio

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        results = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            score = float(person_scores[idx])
            results.append((PERSON_CLASS_ID, score, (int(x1), int(y1), int(x2), int(y2))))

        results.sort(key=lambda item: item[1], reverse=True)
        return results


# =============================================================================
# 4. 摄像头与线程采集
# =============================================================================


def open_camera(source: Union[str, int], width: int, height: int, fps: int) -> cv2.VideoCapture:
    """
    打开摄像头或视频文件。

    Linux 摄像头推荐传入：/dev/video41、/dev/video43。
    这里默认使用 V4L2，并设置 MJPG，保证双路 1080p 传输带宽更低。
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str) and source.startswith("/dev/video"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        use_v4l2_settings = True
    else:
        cap = cv2.VideoCapture(source)
        use_v4l2_settings = False

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    if use_v4l2_settings:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[视频源] 已打开: {source}")
    print(f"[视频源] 实际分辨率: {real_w}x{real_h}, FPS: {real_fps:.1f}")

    return cap


class LatestFrameCamera:
    """
    后台采集线程。

    主线程做 RKNN 推理时可能比较慢，如果直接 cap.read()，摄像头缓冲可能堆积，
    画面会越来越延迟。这个线程始终读最新帧，主线程只拿当前最新帧。
    """

    def __init__(self, cap: cv2.VideoCapture, name: str):
        self.cap = cap
        self.name = name
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _worker(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ok = bool(ret and frame is not None)
                if self.ok:
                    self.frame = frame
            if not self.ok:
                time.sleep(0.005)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None
            # 返回 copy，避免主线程处理时后台线程覆盖同一块内存。
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass


class FPSCounter:
    """简单 FPS 统计器。"""

    def __init__(self):
        self.t0 = time.time()
        self.count = 0
        self.fps = 0.0

    def update(self) -> float:
        self.count += 1
        now = time.time()
        dt = now - self.t0
        if dt >= 1.0:
            self.fps = self.count / dt
            self.count = 0
            self.t0 = now
        return self.fps


# =============================================================================
# 5. 显示与输出工具
# =============================================================================


def draw_raw_results(img: np.ndarray, results, title: str) -> np.ndarray:
    """在原始画面上画 RKNN 检测框。"""
    vis = img.copy()
    for _, score, (x1, y1, x2, y2) in results:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {score:.2f}"
        cv2.putText(vis, label, (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


def detections_to_jsonable(results) -> List[Dict]:
    """把 detect() 的结果转换成可写 JSON 的字典。"""
    out = []
    for class_id, score, (x1, y1, x2, y2) in results:
        out.append({
            "class_id": int(class_id),
            "class_name": "person",
            "score": float(score),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "bottom_center": [float((x1 + x2) / 2.0), float(y2)],
        })
    return out


# =============================================================================
# 6. 主程序：双路 raw 检测测试
# =============================================================================


def main():
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    parser = argparse.ArgumentParser(description="双路 RKNN raw 人体检测测试脚本")

    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头，例如 /dev/video41")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头，例如 /dev/video43")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="RKNN 模型路径")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="labels.txt 路径")
    parser.add_argument("--width", type=int, default=1920, help="采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="采集高度")
    parser.add_argument("--fps", type=int, default=30, help="采集 FPS")
    parser.add_argument("--conf", type=float, default=0.25, help="person 置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS 阈值")
    parser.add_argument("--left-core", default="0", help="左路 RKNN NPU core：0/1/2/all/auto")
    parser.add_argument("--right-core", default="1", help="右路 RKNN NPU core：0/1/2/all/auto")
    parser.add_argument("--bgr-input", action="store_true", help="模型输入为 BGR；默认会 BGR->RGB")
    parser.add_argument("--display", action="store_true", help="显示左右原始检测画面")
    parser.add_argument("--display-scale", type=float, default=0.35, help="显示缩放比例")
    parser.add_argument("--output-jsonl", default="", help="可选：输出 raw 检测结果 JSONL")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示无限")

    args = parser.parse_args()

    left_detector = None
    right_detector = None
    left_cam = None
    right_cam = None
    jsonl_fp = None

    try:
        left_detector = PersonDetector(
            model_path=args.model,
            labels_path=args.labels,
            obj_thresh=args.conf,
            nms_thresh=args.nms,
            use_rgb=not args.bgr_input,
            core=args.left_core,
            name="left-rknn",
        )
        right_detector = PersonDetector(
            model_path=args.model,
            labels_path=args.labels,
            obj_thresh=args.conf,
            nms_thresh=args.nms,
            use_rgb=not args.bgr_input,
            core=args.right_core,
            name="right-rknn",
        )

        cap_left = open_camera(args.left_device, args.width, args.height, args.fps)
        cap_right = open_camera(args.right_device, args.width, args.height, args.fps)
        left_cam = LatestFrameCamera(cap_left, "left").start()
        right_cam = LatestFrameCamera(cap_right, "right").start()

        if args.output_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
            jsonl_fp = open(args.output_jsonl, "w", encoding="utf-8")
            print(f"[输出] raw 检测结果写入: {args.output_jsonl}")

        fps_counter = FPSCounter()
        frame_id = 0

        print("[信息] 开始双路 RKNN raw 检测，按 q / ESC 退出")

        while not STOP_REQUESTED:
            ok_l, left_frame = left_cam.read_latest()
            ok_r, right_frame = right_cam.read_latest()
            if not ok_l or not ok_r:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            left_results = left_detector.detect(left_frame)
            right_results = right_detector.detect(right_frame)
            t1 = time.perf_counter()

            fps = fps_counter.update()

            record = {
                "frame": int(frame_id),
                "time": time.time(),
                "left": detections_to_jsonable(left_results),
                "right": detections_to_jsonable(right_results),
                "infer_ms": (t1 - t0) * 1000.0,
            }

            if jsonl_fp is not None:
                jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fp.flush()

            if frame_id % 10 == 0:
                print(
                    f"[FRAME {frame_id}] fps={fps:.1f} infer={(t1 - t0) * 1000.0:.1f}ms "
                    f"left={len(left_results)} right={len(right_results)}"
                )

            if args.display:
                left_vis = draw_raw_results(left_frame, left_results, "LEFT RAW")
                right_vis = draw_raw_results(right_frame, right_results, "RIGHT RAW")
                h = min(left_vis.shape[0], right_vis.shape[0])
                show = np.hstack([left_vis[:h], right_vis[:h]])
                if args.display_scale != 1.0:
                    show = cv2.resize(show, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
                cv2.imshow("dual rknn raw detect", show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()
        if left_cam is not None:
            left_cam.stop()
        if right_cam is not None:
            right_cam.stop()
        try:
            cap_left.release()
            cap_right.release()
        except Exception:
            pass
        if left_detector is not None:
            left_detector.close()
        if right_detector is not None:
            right_detector.close()
        cv2.destroyAllWindows()
        print("[信息] 程序退出")


if __name__ == "__main__":
    main()
