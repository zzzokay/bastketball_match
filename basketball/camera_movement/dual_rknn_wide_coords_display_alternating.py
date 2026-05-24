#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dual_rknn_wide_coords_display_alternating.py

用途：
    在 RK3588 / ELF2 Linux 上进行“双路摄像头 + 单 RKNN 交替检测 + 坐标映射 + 宽图显示”。

核心目标：
    1. 双路摄像头采集原始 1920x1080 图像。
    2. 每次只对一路摄像头做 RKNN 人体检测，左右交替执行，避免双路 RKNN 同时抢 NPU。
    3. 缓存最近一次左路和右路检测结果。
    4. 使用 stereo_rectify_maps_wide_good.npz 的 remap 表，把 raw 原图检测框映射到 rectified 坐标。
    5. 再根据 stitch_params_good.npz 和你调好的 right_x_shift/right_y_shift，把 rectified 坐标映射到完整宽幅图坐标。
    6. 生成完整宽幅图，并在宽幅图上画出每个人的框、底部中心点和坐标。
    7. 支持 --detect-interval 控制检测间隔。
    8. 支持 --smooth 对宽图坐标做指数平滑，减少运镜目标点抖动。

为什么要“左右交替检测”：
    你之前测试双路 RKNN 并行线程后，主循环仍然只有约 6 FPS，总耗时约 160ms。
    说明同进程双 RKNN 线程没有真正带来收益，或者 NPU/内存/调度资源存在竞争。
    因此这个脚本采用“单 RKNN 实例 + 左右交替检测”的方式：
        第一次检测左路
        第二次检测右路
        第三次检测左路
        第四次检测右路
        ...

    这样每次主循环最多只做一次 RKNN 推理，不会等待两次 detect() 累加。

注意：
    1. 本脚本不保存视频。
    2. 屏幕显示的是完整宽幅拼接图，检测框是由 raw 原图检测结果映射过去的。
    3. 左路默认会屏蔽 overlap 区域，只检测左边非重叠区域；
       右路负责 overlap + 右侧区域，避免同一个人在重叠区域重复输出。
    4. 因为你的拼接算法存在旋转/变形，不能用简单缩放或平移映射坐标；
       本脚本使用 remap 表构造 raw -> rectified 的近似反查表。

典型运行命令：
    cd /home/elf/work/basketball/camera_movement

    python3 dual_rknn_wide_coords_display_alternating.py \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --model /home/elf/work/basketball/model/best_2.rknn \
        --labels /home/elf/work/basketball/model/labels.txt \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
        --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
        --width 1920 \
        --height 1080 \
        --fps 30 \
        --conf 0.25 \
        --nms 0.45 \
        --rknn-core -1 \
        --runtime-seam-x 150 \
        --runtime-blend-width 40 \
        --runtime-right-x-shift 30 \
        --runtime-right-y-shift -5 \
        --detect-interval 1 \
        --smooth 0.65 \
        --display-scale 0.25

按键：
    q / ESC : 退出
    s       : 保存当前宽幅调试 JPG
    l       : 开启/关闭左路 overlap 屏蔽显示逻辑，仅影响后续检测输入
"""

import argparse
import json
import os
import select
import signal
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# RK3588 / Mali 平台上 OpenCV 有时会尝试启用 OpenCL，
# 可能出现 CL_INVALID_BINARY 或额外开销。
# 必须在 import cv2 之前设置。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np
from rknnlite.api import RKNNLite

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# =============================================================================
# 0. 全局退出标志
# =============================================================================

STOP_REQUESTED = False


def handle_exit_signal(signum, frame):
    """捕获 Ctrl+C / kill 信号，让程序尽量正常释放摄像头和 RKNN 资源。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[信息] 收到退出信号，当前帧结束后退出。")


# =============================================================================
# 1. 默认路径和常量
# =============================================================================

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"

DEFAULT_MODEL = "/home/elf/work/basketball/model/best_2.rknn"
DEFAULT_LABELS = "/home/elf/work/basketball/model/labels.txt"

DEFAULT_MAP_FILE = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz"
DEFAULT_STITCH_PARAM = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz"

DEFAULT_SAVE_DIR = "/home/elf/work/basketball/camera_movement/debug_wide"

# 你的 RKNN 模型输入尺寸。当前 best_2.rknn 为 640x640。
MODEL_INPUT_SIZE_DEFAULT = 640

# 只检测 person 类。你的 best_2.rknn 输出 shape 为 (1, 5, 8400)，通常表示 1 类：
# [cx, cy, w, h, person_score]
PERSON_CLASS_ID = 0


# =============================================================================
# 2. 数据结构
# =============================================================================

@dataclass
class RectifyMaps:
    """
    raw -> rectified 使用的 remap 表。

    注意：
        npz 文件中的 left_rect_map1/2、right_rect_map1/2 是给 cv2.remap 用的。
        它们的方向是：
            rectified 输出坐标 -> raw 原图采样坐标

        也就是：
            left_rectified = cv2.remap(left_raw, left_rect_map1, left_rect_map2)

        但我们现在要把 raw 检测框坐标映射到 rectified，
        所以后面会基于这些 map 构造一个近似反查表：
            raw 坐标 -> rectified 坐标
    """

    raw_image_size: Tuple[int, int]
    rectified_size: Tuple[int, int]

    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


@dataclass
class InverseRectifyMaps:
    """
    raw -> rectified 的近似反查表。

    inv_x / inv_y 的尺寸是 raw 图尺寸：
        inv_x[raw_y, raw_x] = rectified_x
        inv_y[raw_y, raw_x] = rectified_y

    如果某些 raw 像素没有直接反查值，会存 NaN。
    映射单个点时，会在附近小范围搜索最近的有效反查值。
    """

    left_inv_x: np.ndarray
    left_inv_y: np.ndarray
    right_inv_x: np.ndarray
    right_inv_y: np.ndarray


@dataclass
class StitchParams:
    """
    离线拼接参数，来自 stitch_params_good.npz。

    宽幅图逻辑结构：
        [left_keep][overlap][right_keep]

    左路：
        left_keep    = left rectified 中非重叠区域
        left_overlap = left rectified 中重叠区域

    右路：
        right_overlap = right rectified 中重叠区域
        right_keep    = right rectified 中非重叠区域

    你的实时微调 right_x_shift/right_y_shift 会作用在右路取样坐标上。
    """

    overlap_px: int
    vertical_offset: int
    blend_width: int

    left_y1: int
    left_y2: int
    right_y1: int
    right_y2: int

    left_keep_x1: int
    left_keep_x2: int
    left_overlap_x1: int
    left_overlap_x2: int

    right_overlap_x1: int
    right_overlap_x2: int
    right_keep_x1: int
    right_keep_x2: int

    output_width: int
    output_height: int


@dataclass
class WideDetection:
    """
    一个映射到完整宽幅图上的检测结果。

    source:
        "left" 或 "right"，表示来自哪一路摄像头。

    raw_bbox:
        原始摄像头图像上的检测框。

    rect_bottom:
        人物底部中心点在 rectified 坐标系中的位置。

    wide_bottom:
        人物底部中心点在完整宽幅图中的位置。

    wide_bbox:
        检测框若干采样点映射到宽幅图后得到的近似外接框。
        由于 remap 可能有旋转/变形，这个框是近似轴对齐框。
    """

    source: str
    score: float
    raw_bbox: Tuple[int, int, int, int]
    rect_bottom: Tuple[float, float]
    wide_bottom: Tuple[float, float]
    wide_bbox: Tuple[float, float, float, float]
    track_id: int = -1


# =============================================================================
# 3. 通用工具函数
# =============================================================================

def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    """从 npz 字段中读取图像尺寸，返回 (width, height)。"""
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 字段格式不正确: {value}")
    return int(flat[0]), int(flat[1])


def get_npz_int(data: np.lib.npyio.NpzFile, key: str, default: Optional[int] = None) -> int:
    """从 npz 中读取 int 字段。"""
    if key not in data.files:
        if default is None:
            raise RuntimeError(f"npz 缺少字段: {key}")
        return int(default)
    return int(np.array(data[key]).reshape(-1)[0])


def current_timestamp() -> str:
    """生成文件名安全的时间戳。"""
    return time.strftime("%Y%m%d_%H%M%S")


def read_terminal_key() -> int:
    """
    从终端非阻塞读取按键。

    OpenCV 窗口有时候没有焦点，cv2.waitKey 读不到按键；
    这个函数允许你在终端输入 q/s/l 后回车。
    """
    try:
        if not sys.stdin.isatty():
            return 255

        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return 255

        line = sys.stdin.readline().strip()
        if not line:
            return 255

        return ord(line[0])
    except Exception:
        return 255


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活，用于处理未归一化的 YOLO 输出。"""
    return 1.0 / (1.0 + np.exp(-x))


def load_labels(labels_path: str) -> List[str]:
    """加载 labels.txt。"""
    path = Path(labels_path)
    if not path.exists():
        print(f"[警告] 标签文件不存在: {labels_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    return labels


def letterbox(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, float, float]:
    """
    YOLO 标准预处理：等比缩放 + 灰色填充到 target_size x target_size。

    返回：
        canvas:
            送入模型的图像。

        ratio:
            原图到 canvas 中有效图像区域的缩放比例。

        pad_w / pad_h:
            水平/垂直方向填充量，用于后处理把坐标映射回原图。
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

    boxes:
        [N, 4]，格式 x1,y1,x2,y2。

    scores:
        [N]，置信度。

    返回：
        保留的索引列表。
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

        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh

        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        remain = np.where(iou <= iou_threshold)[0]
        order = order[remain + 1]

    return keep


# =============================================================================
# 4. RKNN 人体检测器
# =============================================================================

def get_core_mask(core_id: int):
    """
    将命令行传入的 core_id 转成 RKNNLite core_mask。

    core_id:
         0 -> NPU_CORE_0
         1 -> NPU_CORE_1
         2 -> NPU_CORE_2
        -1 -> NPU_CORE_0_1_2，使用全部 NPU core

    对本脚本来说，因为每次只跑一次 RKNN 推理，默认 -1 使用全部 NPU core。
    """
    if core_id == 0:
        return RKNNLite.NPU_CORE_0
    if core_id == 1:
        return RKNNLite.NPU_CORE_1
    if core_id == 2:
        return RKNNLite.NPU_CORE_2
    return RKNNLite.NPU_CORE_0_1_2


class PersonDetector:
    """
    基于 RKNNLite 的 person 检测器。

    这个类的解析逻辑兼容两种常见 YOLO 输出：
        1. (1, 84, 8400)：COCO 80 类输出，person 类 id=0
        2. (1, 5, 8400) ：单类别 person 输出，[cx,cy,w,h,score]

    detect() 返回 raw 原始图像坐标：
        [(class_id, score, (x1,y1,x2,y2)), ...]
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        obj_thresh: float,
        nms_thresh: float,
        input_size: int = 640,
        core_id: int = -1,
        use_rgb: bool = True,
        name: str = "rknn",
    ):
        self.name = name
        self.model_path = model_path
        self.labels = load_labels(labels_path)
        self.obj_thresh = float(obj_thresh)
        self.nms_thresh = float(nms_thresh)
        self.input_size = int(input_size)
        self.core_id = int(core_id)
        self.use_rgb = bool(use_rgb)

        self._printed_output_shape = False
        self._printed_score_debug = False

        self.rknn = RKNNLite()

        print(f"[{self.name}] 加载 RKNN 模型: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"[{self.name}] 加载 RKNN 模型失败: ret={ret}")

        core_mask = get_core_mask(self.core_id)
        print(f"[{self.name}] 初始化 RKNN runtime: core_id={self.core_id}")

        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            print(f"[{self.name}] 指定 core 初始化失败，尝试默认 init_runtime()")
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"[{self.name}] RKNN runtime 初始化失败: ret={ret}")

        print(f"[{self.name}] 模型加载成功，输入尺寸: {self.input_size}x{self.input_size}")
        print(f"[{self.name}] 标签数量: {len(self.labels)}")

    def close(self) -> None:
        """释放 RKNN 资源。"""
        try:
            self.rknn.release()
        except Exception:
            pass

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        对单帧 BGR 图像做人检测。

        返回：
            [(class_id, score, (x1,y1,x2,y2)), ...]
        """
        orig_h, orig_w = image_bgr.shape[:2]

        canvas, ratio, pad_w, pad_h = letterbox(image_bgr, self.input_size)

        if self.use_rgb:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        blob = np.expand_dims(canvas, axis=0)
        blob = np.ascontiguousarray(blob)

        try:
            outputs = self.rknn.inference(inputs=[blob], data_format=["nhwc"])
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

        # 兼容 (C, N) 和 (N, C)
        # 例如 (5,8400) / (84,8400) 转成 (8400,5)/(8400,84)
        if pred.shape[0] < pred.shape[1] and pred.shape[0] >= 5:
            output_2d = pred.T
        elif pred.shape[1] >= 5:
            output_2d = pred
        else:
            print(f"[{self.name}] 无法解析的 YOLO 输出 shape: {output.shape}")
            return []

        num_classes = output_2d.shape[1] - 4
        if num_classes <= 0:
            print(f"[{self.name}] 类别数异常: {num_classes}")
            return []

        boxes_xywh = output_2d[:, :4].astype(np.float32)
        cls_scores = output_2d[:, 4:].astype(np.float32)

        # 有些模型输出已经是 0~1 概率；有些是 logits。
        # 如果范围明显超过 0~1，就做 sigmoid。
        if cls_scores.size == 0:
            return []

        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)

        # 只取 person 类。
        # 单类别模型时 num_classes=1，索引 0 就是 person。
        if PERSON_CLASS_ID >= cls_scores.shape[1]:
            return []

        person_scores = cls_scores[:, PERSON_CLASS_ID]

        mask = person_scores > self.obj_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        person_scores = person_scores[mask]

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = nms(boxes, person_scores, self.nms_thresh)

        # 坐标从 letterbox 输入空间还原到原图空间
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes[:, :4] /= ratio

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        results = []
        for k in keep:
            x1, y1, x2, y2 = boxes[k]
            score = float(person_scores[k])
            results.append(
                (
                    PERSON_CLASS_ID,
                    score,
                    (int(x1), int(y1), int(x2), int(y2)),
                )
            )

        results.sort(key=lambda item: item[1], reverse=True)
        return results


# =============================================================================
# 5. 摄像头采集
# =============================================================================

def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """
    打开 V4L2 摄像头。

    设置 MJPG：
        双路 1920x1080@30fps 时，MJPG 比 YUYV 更省 USB 带宽。
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = float(cap.get(cv2.CAP_PROP_FPS))

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))

    print(f"[视频源] 已打开: {device}")
    print(f"[视频源] 实际分辨率: {real_w}x{real_h}, FPS: {real_fps:.1f}, FOURCC: {fourcc_str}")

    return cap


def grab_pair(cap_left: cv2.VideoCapture, cap_right: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    同步读取左右帧。

    使用 grab + retrieve：
        先让两个摄像头各自抓取一帧，再解码取出；
        比直接 read 左再 read 右的时间差更小。
    """
    ok_l = cap_left.grab()
    ok_r = cap_right.grab()

    if not ok_l or not ok_r:
        return False, None, None

    ret_l, left = cap_left.retrieve()
    ret_r, right = cap_right.retrieve()

    if not ret_l or not ret_r or left is None or right is None:
        return False, None, None

    return True, left, right



class LatestFrameCamera:
    """
    后台采集摄像头线程。

    为什么需要它：
        如果主循环里直接执行：
            grab_pair(cap_left, cap_right)

        那么主循环会被摄像头读取阻塞。
        你当前日志里：
            infer ≈ 55ms
            total ≈ 145ms

        说明除了 RKNN 推理外，还有接近 90ms 花在其它地方。
        在双 USB 摄像头场景中，最常见就是同步读取阻塞。

    这个类的做法：
        1. 每个摄像头一个线程。
        2. 线程不断读取最新帧。
        3. 主循环只取最近一帧，不等待摄像头。
        4. 如果主循环处理慢，旧帧自动被新帧覆盖，避免延迟堆积。

    注意：
        这里不保存每一帧，只保存 latest_frame。
        这是实时运镜场景更合理的策略。
    """

    def __init__(self, device: str, width: int, height: int, fps: int, name: str):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name

        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_index = -1
        self.latest_time = 0.0

        self.read_fail_count = 0

    def start(self):
        """
        打开摄像头并启动后台采集线程。
        """
        self.cap = open_camera(self.device, self.width, self.height, self.fps)

        self.thread = threading.Thread(
            target=self._capture_loop,
            name=f"{self.name}-capture",
            daemon=True,
        )
        self.thread.start()
        return self

    def _capture_loop(self):
        """
        后台采集循环。

        使用 cap.read()：
            对 MJPG 摄像头来说，read() 内部完成 grab + decode。
            这个线程独立运行，所以即使 read() 阻塞，也不会卡住主循环。
        """
        idx = 0

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.read_fail_count += 1
                time.sleep(0.005)
                continue

            now = time.time()

            with self.lock:
                self.latest_frame = frame
                self.latest_index = idx
                self.latest_time = now

            idx += 1

    def wait_first_frame(self, timeout: float = 3.0) -> bool:
        """
        等待第一帧到来。

        返回：
            True  : 已经拿到第一帧
            False : 超时
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                ok = self.latest_frame is not None
            if ok:
                return True
            time.sleep(0.01)
        return False

    def get_latest(self):
        """
        获取最近一帧。

        返回：
            frame_index, frame, timestamp

        注意：
            这里不 copy 图像，减少 1920x1080 大图复制开销。
            后台线程下一次 read() 会得到新的 ndarray，通常不会改写旧 ndarray。
        """
        with self.lock:
            if self.latest_frame is None:
                return -1, None, 0.0
            return self.latest_index, self.latest_frame, self.latest_time

    def stop(self):
        """
        停止采集线程并释放摄像头。
        """
        self.stop_event.set()

        if self.thread is not None:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def get_latest_pair(cam_left: LatestFrameCamera, cam_right: LatestFrameCamera):
    """
    从两个后台采集线程中取最新左右帧。

    返回：
        ok, left_frame, right_frame
    """
    li, left, lt = cam_left.get_latest()
    ri, right, rt = cam_right.get_latest()

    if left is None or right is None:
        return False, None, None

    return True, left, right


# =============================================================================
# 6. 加载 maps / stitch 参数
# =============================================================================


def convert_maps_for_fast_remap(map1: np.ndarray, map2: np.ndarray):
    """
    将 OpenCV remap 使用的 map 转成更快的 fixed-point map。

    为什么这里要写得比较兼容：
        OpenCV 的 remap map 常见有三种格式：

        1. 已经是 fixed-point：
            map1: int16, shape = (H, W, 2)
            map2: uint16/int16, shape = (H, W)
            这种不需要再 convertMaps。

        2. 两张 float32 单通道 map：
            map1: float32, shape = (H, W)
            map2: float32, shape = (H, W)
            这种可以：
                cv2.convertMaps(map1, map2, cv2.CV_16SC2)

        3. 一张 float32 双通道 map：
            map1: float32, shape = (H, W, 2)
            map2: None 或无意义
            这种必须：
                cv2.convertMaps(map1, None, cv2.CV_16SC2)

    你这次报错的原因大概率就是第 3 种：
        map1 是 CV_32FC2，但是代码仍然把 map2 一起传入了 convertMaps。
    """

    # 有些 npz 里 map2 可能是空数组、None-like，统一转成 None 处理。
    if map2 is not None:
        try:
            if np.asarray(map2).size == 0:
                map2 = None
        except Exception:
            map2 = None

    # ------------------------------------------------------------
    # 情况 1：已经是 fixed-point map，直接返回
    # ------------------------------------------------------------
    if (
        map1 is not None
        and map1.dtype == np.int16
        and map1.ndim == 3
        and map1.shape[2] == 2
        and map2 is not None
        and map2.dtype in (np.uint16, np.int16)
    ):
        return map1, map2

    # ------------------------------------------------------------
    # 情况 2：map1 是 float32 双通道 CV_32FC2
    # OpenCV 要求第二个参数必须是 None
    # ------------------------------------------------------------
    if (
        map1 is not None
        and map1.ndim == 3
        and map1.shape[2] == 2
    ):
        return cv2.convertMaps(
            map1.astype(np.float32),
            None,
            cv2.CV_16SC2,
        )

    # ------------------------------------------------------------
    # 情况 3：map1/map2 是 float32 单通道
    # ------------------------------------------------------------
    if (
        map1 is not None
        and map2 is not None
        and map1.ndim == 2
        and map2.ndim == 2
    ):
        return cv2.convertMaps(
            map1.astype(np.float32),
            map2.astype(np.float32),
            cv2.CV_16SC2,
        )

    # ------------------------------------------------------------
    # 兜底：如果格式不认识，打印出来方便定位
    # ------------------------------------------------------------
    raise RuntimeError(
        "不支持的 remap map 格式: "
        f"map1 dtype={getattr(map1, 'dtype', None)}, shape={getattr(map1, 'shape', None)}, "
        f"map2 dtype={getattr(map2, 'dtype', None)}, shape={getattr(map2, 'shape', None)}"
    )


def load_rectify_maps(map_file: str) -> Tuple[RectifyMaps, RectifyMaps]:
    """
    加载 remap 文件。

    返回两个对象：
        maps_float:
            保存 float32 map，用于构造反查表。

        maps_fast:
            保存 fixed-point map，用于实时 cv2.remap 生成宽幅图。
    """
    if not os.path.exists(map_file):
        raise RuntimeError(f"找不到 map 文件: {map_file}")

    data = np.load(map_file)

    required = [
        "left_rect_map1",
        "left_rect_map2",
        "right_rect_map1",
        "right_rect_map2",
        "raw_image_size",
    ]

    for key in required:
        if key not in data.files:
            raise RuntimeError(f"map 文件缺少字段: {key}")

    raw_image_size = parse_image_size(data["raw_image_size"])

    if "rectified_size" in data.files:
        rectified_size = parse_image_size(data["rectified_size"])
    else:
        rectified_size = (
            int(data["left_rect_map1"].shape[1]),
            int(data["left_rect_map1"].shape[0]),
        )

    left_m1_f = data["left_rect_map1"].astype(np.float32)
    left_m2_f = data["left_rect_map2"].astype(np.float32)
    right_m1_f = data["right_rect_map1"].astype(np.float32)
    right_m2_f = data["right_rect_map2"].astype(np.float32)

    maps_float = RectifyMaps(
        raw_image_size=raw_image_size,
        rectified_size=rectified_size,
        left_map1=left_m1_f,
        left_map2=left_m2_f,
        right_map1=right_m1_f,
        right_map2=right_m2_f,
    )

    left_m1_fast, left_m2_fast = convert_maps_for_fast_remap(left_m1_f, left_m2_f)
    right_m1_fast, right_m2_fast = convert_maps_for_fast_remap(right_m1_f, right_m2_f)

    maps_fast = RectifyMaps(
        raw_image_size=raw_image_size,
        rectified_size=rectified_size,
        left_map1=left_m1_fast,
        left_map2=left_m2_fast,
        right_map1=right_m1_fast,
        right_map2=right_m2_fast,
    )

    print("[信息] 已加载 remap 文件:")
    print(f"  map_file       : {map_file}")
    print(f"  raw_image_size : {raw_image_size}")
    print(f"  rectified_size : {rectified_size}")

    return maps_float, maps_fast


def load_stitch_params(stitch_param_file: str) -> StitchParams:
    """加载 stitch_params_good.npz。"""
    if not os.path.exists(stitch_param_file):
        raise RuntimeError(f"找不到 stitch 参数文件: {stitch_param_file}")

    data = np.load(stitch_param_file)
    overlap_px = get_npz_int(data, "overlap_px")

    params = StitchParams(
        overlap_px=overlap_px,
        vertical_offset=get_npz_int(data, "vertical_offset"),
        blend_width=get_npz_int(data, "blend_width", default=overlap_px),

        left_y1=get_npz_int(data, "left_y1"),
        left_y2=get_npz_int(data, "left_y2"),
        right_y1=get_npz_int(data, "right_y1"),
        right_y2=get_npz_int(data, "right_y2"),

        left_keep_x1=get_npz_int(data, "left_keep_x1"),
        left_keep_x2=get_npz_int(data, "left_keep_x2"),
        left_overlap_x1=get_npz_int(data, "left_overlap_x1"),
        left_overlap_x2=get_npz_int(data, "left_overlap_x2"),

        right_overlap_x1=get_npz_int(data, "right_overlap_x1"),
        right_overlap_x2=get_npz_int(data, "right_overlap_x2"),
        right_keep_x1=get_npz_int(data, "right_keep_x1"),
        right_keep_x2=get_npz_int(data, "right_keep_x2"),

        output_width=get_npz_int(data, "output_width"),
        output_height=get_npz_int(data, "output_height"),
    )

    print("[信息] 已加载拼接参数:")
    print(f"  overlap_px      : {params.overlap_px}")
    print(f"  vertical_offset : {params.vertical_offset}")
    print(f"  blend_width     : {params.blend_width}")
    print(f"  output_size     : {params.output_width} x {params.output_height}")
    print(f"  left_y          : {params.left_y1} -> {params.left_y2}")
    print(f"  right_y         : {params.right_y1} -> {params.right_y2}")

    return params


# =============================================================================
# 7. 构造 raw -> rectified 反查表
# =============================================================================


def split_float_remap_xy(map1: np.ndarray, map2: np.ndarray):
    """
    将 OpenCV remap map 统一拆成 raw_x / raw_y 两张 float32 单通道表。

    背景：
        你的 stereo_rectify_maps_wide_good.npz 里，map 可能是两种格式：

        1. 双通道格式：
            map1.shape = (H, W, 2)
            map1[:, :, 0] 是 raw_x
            map1[:, :, 1] 是 raw_y
            map2 可能为空或者无意义

        2. 单通道格式：
            map1.shape = (H, W)
            map2.shape = (H, W)
            map1 是 raw_x
            map2 是 raw_y

    build_one_inverse_map() 需要的是单通道 raw_x/raw_y，
    所以这里统一转换。
    """
    if map1 is None:
        raise RuntimeError("map1 is None")

    # 情况 1：CV_32FC2 / 双通道 map
    if map1.ndim == 3 and map1.shape[2] == 2:
        raw_x = map1[:, :, 0].astype(np.float32)
        raw_y = map1[:, :, 1].astype(np.float32)
        return raw_x, raw_y

    # 情况 2：CV_32FC1 + CV_32FC1 / 两张单通道 map
    if map1.ndim == 2 and map2 is not None and map2.ndim == 2:
        raw_x = map1.astype(np.float32)
        raw_y = map2.astype(np.float32)
        return raw_x, raw_y

    raise RuntimeError(
        "不支持的 float remap map 格式: "
        f"map1 dtype={getattr(map1, 'dtype', None)}, shape={getattr(map1, 'shape', None)}, "
        f"map2 dtype={getattr(map2, 'dtype', None)}, shape={getattr(map2, 'shape', None)}"
    )


def build_one_inverse_map(
    map1: np.ndarray,
    map2: np.ndarray,
    raw_size,
    name: str = "",
):
    """
    构造 raw -> rectified 的反查表。

    已知：
        OpenCV remap 的 map 是 rectified -> raw：
            rectified(xr, yr) 从 raw(map_x[yr, xr], map_y[yr, xr]) 取像素

    但我们的检测框是在 raw 图上的：
        YOLO 检测得到 raw 点 (x_raw, y_raw)

    所以要构造一个近似反查：
        raw 点 -> rectified 点

    方法：
        遍历每一个 rectified 像素，看它来自 raw 的哪个位置；
        然后把这个关系反向写到 inv_x/inv_y 中。

    注意：
        这是近似查表，不是严格数学逆变换。
        对人物底部中心点和 bbox 角点映射已经够用。
    """
    raw_w, raw_h = int(raw_size[0]), int(raw_size[1])

    # 兼容双通道 map 和单通道 map
    raw_x, raw_y = split_float_remap_xy(map1, map2)

    rect_h, rect_w = raw_x.shape[:2]

    inv_x = np.full((raw_h, raw_w), -1, dtype=np.float32)
    inv_y = np.full((raw_h, raw_w), -1, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(
        np.arange(rect_w, dtype=np.float32),
        np.arange(rect_h, dtype=np.float32),
    )

    valid = (
        (raw_x >= 0) & (raw_x < raw_w) &
        (raw_y >= 0) & (raw_y < raw_h)
    )

    # raw 坐标四舍五入到整数像素，作为反查表索引
    rx = np.rint(raw_x[valid]).astype(np.int32)
    ry = np.rint(raw_y[valid]).astype(np.int32)

    # 再做一次安全裁剪，避免极少数 round 后越界
    rx = np.clip(rx, 0, raw_w - 1)
    ry = np.clip(ry, 0, raw_h - 1)

    inv_x[ry, rx] = grid_x[valid]
    inv_y[ry, rx] = grid_y[valid]

    # 简单填洞：
    # 有些 raw 像素没有被任何 rectified 像素直接命中，inv 会是 -1。
    # 这里用 inpaint 对 inv_x/inv_y 填补，避免检测点落在空洞时报无效。
    mask = (inv_x < 0) | (inv_y < 0)
    if np.any(mask):
        valid_mask = (~mask).astype(np.uint8) * 255
        hole_mask = mask.astype(np.uint8) * 255

        # inpaint 要求输入 float32/uint8 单通道，这里用 float32 可以。
        inv_x_filled = cv2.inpaint(inv_x, hole_mask, 3, cv2.INPAINT_NS)
        inv_y_filled = cv2.inpaint(inv_y, hole_mask, 3, cv2.INPAINT_NS)

        inv_x[mask] = inv_x_filled[mask]
        inv_y[mask] = inv_y_filled[mask]

    print(f"[信息] {name} 反查表完成: raw {raw_w}x{raw_h} -> rect {rect_w}x{rect_h}")
    return inv_x, inv_y


def load_or_build_inverse_maps(
    maps_float: RectifyMaps,
    map_file: str,
    cache_file: Optional[str],
    rebuild: bool = False,
) -> InverseRectifyMaps:
    """
    加载或构造 raw->rectified 反查表。

    缓存原因：
        反查表构造只需要做一次。
        保存到 npz 后，下次启动可以直接加载，减少启动时间。
    """
    if cache_file is None:
        base = Path(map_file)
        cache_file = str(base.with_name(base.stem + "_inverse_raw_to_rect.npz"))

    if (not rebuild) and os.path.exists(cache_file):
        print(f"[信息] 加载 inverse map 缓存: {cache_file}")
        data = np.load(cache_file)
        return InverseRectifyMaps(
            left_inv_x=data["left_inv_x"].astype(np.float32),
            left_inv_y=data["left_inv_y"].astype(np.float32),
            right_inv_x=data["right_inv_x"].astype(np.float32),
            right_inv_y=data["right_inv_y"].astype(np.float32),
        )

    left_inv_x, left_inv_y = build_one_inverse_map(
        maps_float.left_map1,
        maps_float.left_map2,
        maps_float.raw_image_size,
        "left",
    )

    right_inv_x, right_inv_y = build_one_inverse_map(
        maps_float.right_map1,
        maps_float.right_map2,
        maps_float.raw_image_size,
        "right",
    )

    print(f"[信息] 保存 inverse map 缓存: {cache_file}")
    np.savez_compressed(
        cache_file,
        left_inv_x=left_inv_x,
        left_inv_y=left_inv_y,
        right_inv_x=right_inv_x,
        right_inv_y=right_inv_y,
    )

    return InverseRectifyMaps(
        left_inv_x=left_inv_x,
        left_inv_y=left_inv_y,
        right_inv_x=right_inv_x,
        right_inv_y=right_inv_y,
    )


def raw_point_to_rectified(
    inv_x: np.ndarray,
    inv_y: np.ndarray,
    raw_x: float,
    raw_y: float,
    search_radius: int = 8,
) -> Optional[Tuple[float, float]]:
    """
    将 raw 原图坐标点映射到 rectified 坐标。

    由于反查表是近似稀疏的，有些 raw 像素可能是 NaN。
    处理方式：
        1. 先检查 raw 点四舍五入后的像素是否有值。
        2. 如果没有，就在 search_radius 范围内找最近的有效反查点。

    返回：
        (rect_x, rect_y)，找不到则返回 None。
    """
    h, w = inv_x.shape[:2]

    ix = int(round(raw_x))
    iy = int(round(raw_y))

    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None

    if np.isfinite(inv_x[iy, ix]) and np.isfinite(inv_y[iy, ix]):
        return float(inv_x[iy, ix]), float(inv_y[iy, ix])

    r = int(max(1, search_radius))

    x1 = max(0, ix - r)
    x2 = min(w, ix + r + 1)
    y1 = max(0, iy - r)
    y2 = min(h, iy + r + 1)

    patch_x = inv_x[y1:y2, x1:x2]
    patch_y = inv_y[y1:y2, x1:x2]

    valid = np.isfinite(patch_x) & np.isfinite(patch_y)

    if not np.any(valid):
        return None

    yy, xx = np.where(valid)
    abs_x = xx + x1
    abs_y = yy + y1

    dist2 = (abs_x - ix) ** 2 + (abs_y - iy) ** 2
    best = int(np.argmin(dist2))

    py = yy[best]
    px = xx[best]

    return float(patch_x[py, px]), float(patch_y[py, px])


# =============================================================================
# 8. 宽幅拼接图生成
# =============================================================================

def remap_rectified_roi_fixed_size(
    raw: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    rect_x1: int,
    rect_y1: int,
    rect_x2: int,
    rect_y2: int,
) -> np.ndarray:
    """
    从 raw 图中 remap 出 rectified 坐标系下的一个 ROI。

    这个函数保证输出固定大小：
        out_w = rect_x2 - rect_x1
        out_h = rect_y2 - rect_y1

    如果 ROI 因 right_x_shift/right_y_shift 越界，会用黑边补齐。
    """
    out_w = int(rect_x2 - rect_x1)
    out_h = int(rect_y2 - rect_y1)

    if out_w <= 0 or out_h <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    map_h, map_w = map1.shape[:2]

    cx1 = max(0, min(map_w, int(rect_x1)))
    cx2 = max(0, min(map_w, int(rect_x2)))
    cy1 = max(0, min(map_h, int(rect_y1)))
    cy2 = max(0, min(map_h, int(rect_y2)))

    if cx2 <= cx1 or cy2 <= cy1:
        return out

    roi = cv2.remap(
        raw,
        map1[cy1:cy2, cx1:cx2],
        map2[cy1:cy2, cx1:cx2],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    dx1 = cx1 - int(rect_x1)
    dy1 = cy1 - int(rect_y1)

    out[dy1:dy1 + roi.shape[0], dx1:dx1 + roi.shape[1]] = roi

    return out


def build_wide_stitch_from_raw(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    maps_fast: RectifyMaps,
    params: StitchParams,
    seam_x: int,
    blend_width: int,
    right_x_shift: int,
    right_y_shift: int,
) -> np.ndarray:
    """
    从左右 raw 图直接生成完整宽幅拼接图。

    这里复用了你之前调好的实时拼接思路：
        1. 不生成完整 left_rect/right_rect。
        2. 直接 remap 拼接所需 ROI。
        3. overlap 左侧用左图，右侧用右图，只在 seam 附近做小范围融合。
        4. 使用 runtime-right-x-shift / runtime-right-y-shift 微调右图取样。
    """
    if (left_raw.shape[1], left_raw.shape[0]) != maps_fast.raw_image_size:
        left_raw = cv2.resize(left_raw, maps_fast.raw_image_size)

    if (right_raw.shape[1], right_raw.shape[0]) != maps_fast.raw_image_size:
        right_raw = cv2.resize(right_raw, maps_fast.raw_image_size)

    output_w = int(params.output_width)
    output_h = int(params.output_height)

    wide = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    left_keep_w = params.left_keep_x2 - params.left_keep_x1
    overlap_w = params.overlap_px
    right_keep_w = params.right_keep_x2 - params.right_keep_x1

    left_keep_start = 0
    left_keep_end = left_keep_w

    overlap_start = left_keep_end
    overlap_end = overlap_start + overlap_w

    right_keep_start = overlap_end
    right_keep_end = right_keep_start + right_keep_w

    left_y1 = params.left_y1
    left_y2 = left_y1 + output_h

    right_y1 = params.right_y1 + right_y_shift
    right_y2 = right_y1 + output_h

    # 1. 左侧非重叠区域
    left_keep = remap_rectified_roi_fixed_size(
        left_raw,
        maps_fast.left_map1,
        maps_fast.left_map2,
        params.left_keep_x1,
        left_y1,
        params.left_keep_x2,
        left_y2,
    )
    wide[:, left_keep_start:left_keep_end] = left_keep

    # 2. overlap 区域：根据 seam_x 和 blend_width 划分
    blend_width = max(1, min(int(blend_width), overlap_w))

    if seam_x < 0:
        seam_x_used = overlap_w // 2
    else:
        seam_x_used = int(seam_x)

    half_blend = blend_width // 2

    seam_x_used = int(np.clip(
        seam_x_used,
        half_blend,
        overlap_w - (blend_width - half_blend),
    ))

    blend_x1 = seam_x_used - half_blend
    blend_x2 = blend_x1 + blend_width

    # overlap 左侧直接用左图，减少重影
    if blend_x1 > 0:
        left_overlap_left = remap_rectified_roi_fixed_size(
            left_raw,
            maps_fast.left_map1,
            maps_fast.left_map2,
            params.left_overlap_x1,
            left_y1,
            params.left_overlap_x1 + blend_x1,
            left_y2,
        )
        wide[:, overlap_start:overlap_start + blend_x1] = left_overlap_left

    # overlap 右侧直接用右图
    if blend_x2 < overlap_w:
        right_overlap_right = remap_rectified_roi_fixed_size(
            right_raw,
            maps_fast.right_map1,
            maps_fast.right_map2,
            params.right_overlap_x1 + blend_x2 + right_x_shift,
            right_y1,
            params.right_overlap_x1 + overlap_w + right_x_shift,
            right_y2,
        )
        wide[:, overlap_start + blend_x2:overlap_end] = right_overlap_right

    # seam 附近做 alpha 融合
    left_blend = remap_rectified_roi_fixed_size(
        left_raw,
        maps_fast.left_map1,
        maps_fast.left_map2,
        params.left_overlap_x1 + blend_x1,
        left_y1,
        params.left_overlap_x1 + blend_x2,
        left_y2,
    )

    right_blend = remap_rectified_roi_fixed_size(
        right_raw,
        maps_fast.right_map1,
        maps_fast.right_map2,
        params.right_overlap_x1 + blend_x1 + right_x_shift,
        right_y1,
        params.right_overlap_x1 + blend_x2 + right_x_shift,
        right_y2,
    )

    alpha = np.linspace(0.0, 1.0, blend_width, dtype=np.float32).reshape(1, blend_width, 1)
    blend = left_blend.astype(np.float32) * (1.0 - alpha) + right_blend.astype(np.float32) * alpha
    blend = np.clip(blend, 0, 255).astype(np.uint8)

    wide[:, overlap_start + blend_x1:overlap_start + blend_x2] = blend

    # 3. 右侧非重叠区域
    right_keep = remap_rectified_roi_fixed_size(
        right_raw,
        maps_fast.right_map1,
        maps_fast.right_map2,
        params.right_keep_x1 + right_x_shift,
        right_y1,
        params.right_keep_x2 + right_x_shift,
        right_y2,
    )
    wide[:, right_keep_start:right_keep_end] = right_keep

    return wide


# =============================================================================
# 9. 左路 overlap 屏蔽 mask
# =============================================================================


def build_left_keep_raw_mask(
    maps_float: RectifyMaps,
    params: StitchParams,
    dilate_iter: int = 3,
) -> np.ndarray:
    """
    生成左路 raw 图上的 left_keep mask。

    目的：
        你希望“重叠区域直接去掉左边重叠部分”，
        因此左路 RKNN 不应该检测 overlap 区域。

    为什么要在 raw 图上生成 mask：
        RKNN 检测是在摄像头原始图 raw 上做的；
        但是 left_keep / overlap 是 rectified 坐标系里的区域。
        所以必须用 remap 表把 rectified 的 left_keep 区域反投到 raw 图上。

    关键点：
        OpenCV remap 的 map 有两种常见格式：

        1. 单通道格式：
            map1.shape = (H, W)
            map2.shape = (H, W)
            map1 是 raw_x
            map2 是 raw_y

        2. 双通道格式：
            map1.shape = (H, W, 2)
            map1[:, :, 0] 是 raw_x
            map1[:, :, 1] 是 raw_y
            map2 可能存在但不能直接当 raw_y 用

        你当前 stereo_rectify_maps_wide_good.npz 使用的是双通道 map，
        所以这里必须通过 split_float_remap_xy() 统一拆成 raw_x/raw_y。
    """
    raw_w, raw_h = maps_float.raw_image_size
    mask = np.zeros((raw_h, raw_w), dtype=np.uint8)

    # left_keep 在 rectified 坐标系中的范围
    x1 = max(0, int(params.left_keep_x1))
    x2 = min(int(maps_float.left_map1.shape[1]), int(params.left_keep_x2))
    y1 = max(0, int(params.left_y1))
    y2 = min(int(maps_float.left_map1.shape[0]), int(params.left_y2))

    if x2 <= x1 or y2 <= y1:
        print("[警告] left_keep ROI 无效，返回全 0 mask")
        return mask

    roi_m1 = maps_float.left_map1[y1:y2, x1:x2]

    # map2 可能为 None，也可能是单通道 map。
    # 如果 map1 是双通道，split_float_remap_xy 会自动忽略 map2。
    if maps_float.left_map2 is not None:
        roi_m2 = maps_float.left_map2[y1:y2, x1:x2]
    else:
        roi_m2 = None

    raw_x_f, raw_y_f = split_float_remap_xy(roi_m1, roi_m2)

    raw_x = np.rint(raw_x_f).astype(np.int32)
    raw_y = np.rint(raw_y_f).astype(np.int32)

    valid = (
        (raw_x >= 0) & (raw_x < raw_w) &
        (raw_y >= 0) & (raw_y < raw_h)
    )

    mask[raw_y[valid], raw_x[valid]] = 255

    # 膨胀几次，填补 remap 离散采样造成的小孔洞。
    # 这样左路检测区域不会因为 mask 小洞被切碎。
    if dilate_iter > 0:
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    print(
        "[信息] left_keep raw mask 完成: "
        f"raw={raw_w}x{raw_h}, keep_rect=({x1},{y1})-({x2},{y2}), "
        f"valid_pixels={int((mask > 0).sum())}"
    )

    return mask


def apply_detection_mask(frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    对检测输入应用 mask。

    mask == 0 的区域填成 114，和 YOLO letterbox 填充色一致。
    这样可以降低被屏蔽区域误检概率。
    """
    if mask is None:
        return frame

    out = frame.copy()

    if mask.shape[:2] != out.shape[:2]:
        mask = cv2.resize(mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)

    out[mask == 0] = (114, 114, 114)
    return out


# =============================================================================
# 10. raw 检测框 -> 宽幅坐标
# =============================================================================

def left_rect_to_wide(
    rect_x: float,
    rect_y: float,
    params: StitchParams,
    allow_left_overlap: bool = False,
) -> Optional[Tuple[float, float]]:
    """
    左路 rectified 坐标 -> 完整宽幅图坐标。

    默认 allow_left_overlap=False：
        左路 overlap 直接丢弃，只保留 left_keep。
    """
    left_keep_w = params.left_keep_x2 - params.left_keep_x1

    wide_y = rect_y - params.left_y1

    if wide_y < 0 or wide_y >= params.output_height:
        return None

    # 左路非重叠区域
    if params.left_keep_x1 <= rect_x < params.left_keep_x2:
        wide_x = rect_x - params.left_keep_x1
        return float(wide_x), float(wide_y)

    # 左路 overlap 区域。按你的需求默认不使用。
    if allow_left_overlap and params.left_overlap_x1 <= rect_x < params.left_overlap_x2:
        local = rect_x - params.left_overlap_x1
        wide_x = left_keep_w + local
        return float(wide_x), float(wide_y)

    return None


def right_rect_to_wide(
    rect_x: float,
    rect_y: float,
    params: StitchParams,
    right_x_shift: int,
    right_y_shift: int,
) -> Optional[Tuple[float, float]]:
    """
    右路 rectified 坐标 -> 完整宽幅图坐标。

    注意右路微调：
        你的拼接代码在 remap 右图时使用：
            rect_x + right_x_shift
            rect_y + right_y_shift

        因此 raw 点反查到的 rectified 坐标要反过来扣除这个 shift，
        才能得到它落在宽幅图中的 local 位置。
    """
    left_keep_w = params.left_keep_x2 - params.left_keep_x1
    overlap_w = params.overlap_px

    overlap_start = left_keep_w
    right_keep_start = left_keep_w + overlap_w

    # 把右图实际 rectified 坐标转回拼接局部坐标使用的“未微调坐标”
    rx = rect_x - right_x_shift
    ry = rect_y - right_y_shift

    wide_y = ry - params.right_y1

    if wide_y < 0 or wide_y >= params.output_height:
        return None

    # 右路 overlap
    if params.right_overlap_x1 <= rx < params.right_overlap_x2:
        local = rx - params.right_overlap_x1
        wide_x = overlap_start + local
        return float(wide_x), float(wide_y)

    # 右路非重叠区域
    if params.right_keep_x1 <= rx < params.right_keep_x2:
        local = rx - params.right_keep_x1
        wide_x = right_keep_start + local
        return float(wide_x), float(wide_y)

    return None


def map_raw_bbox_to_wide(
    source: str,
    bbox: Tuple[int, int, int, int],
    score: float,
    inv_maps: InverseRectifyMaps,
    params: StitchParams,
    right_x_shift: int,
    right_y_shift: int,
    search_radius: int,
    allow_left_overlap: bool = False,
) -> Optional[WideDetection]:
    """
    将一个 raw 检测框映射到完整宽幅图坐标。

    为什么不只映射 bbox 四角：
        因为 remap 可能有旋转/变形。
        只用两个角可能不稳定。

    这里采样：
        四个角
        上边中心
        下边中心
        左边中心
        右边中心
        框中心

    然后把这些点映射到宽幅图，取外接矩形作为 wide_bbox。
    人物底部中心点单独保存为 wide_bottom，后续运镜优先用它。
    """
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bottom_x = cx
    bottom_y = float(y2)

    sample_points = [
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
        (cx, y1),
        (cx, y2),
        (x1, cy),
        (x2, cy),
        (cx, cy),
    ]

    if source == "left":
        inv_x = inv_maps.left_inv_x
        inv_y = inv_maps.left_inv_y
    else:
        inv_x = inv_maps.right_inv_x
        inv_y = inv_maps.right_inv_y

    # 先映射底部中心点。这个点是运镜最重要的目标点。
    rect_bottom = raw_point_to_rectified(
        inv_x,
        inv_y,
        bottom_x,
        bottom_y,
        search_radius=search_radius,
    )

    if rect_bottom is None:
        return None

    if source == "left":
        wide_bottom = left_rect_to_wide(
            rect_bottom[0],
            rect_bottom[1],
            params,
            allow_left_overlap=allow_left_overlap,
        )
    else:
        wide_bottom = right_rect_to_wide(
            rect_bottom[0],
            rect_bottom[1],
            params,
            right_x_shift=right_x_shift,
            right_y_shift=right_y_shift,
        )

    # 如果底部中心点落不到有效宽图区域，直接丢弃这个检测
    if wide_bottom is None:
        return None

    wide_points = []

    for px, py in sample_points:
        rect_pt = raw_point_to_rectified(
            inv_x,
            inv_y,
            px,
            py,
            search_radius=search_radius,
        )

        if rect_pt is None:
            continue

        if source == "left":
            wide_pt = left_rect_to_wide(
                rect_pt[0],
                rect_pt[1],
                params,
                allow_left_overlap=allow_left_overlap,
            )
        else:
            wide_pt = right_rect_to_wide(
                rect_pt[0],
                rect_pt[1],
                params,
                right_x_shift=right_x_shift,
                right_y_shift=right_y_shift,
            )

        if wide_pt is not None:
            wide_points.append(wide_pt)

    if len(wide_points) < 2:
        # 如果可用采样点太少，就围绕底部中心给一个粗略小框，避免程序崩溃
        bx, by = wide_bottom
        wide_bbox = (bx - 20.0, by - 80.0, bx + 20.0, by)
    else:
        xs = [p[0] for p in wide_points]
        ys = [p[1] for p in wide_points]
        wide_bbox = (
            float(np.clip(min(xs), 0, params.output_width - 1)),
            float(np.clip(min(ys), 0, params.output_height - 1)),
            float(np.clip(max(xs), 0, params.output_width - 1)),
            float(np.clip(max(ys), 0, params.output_height - 1)),
        )

    return WideDetection(
        source=source,
        score=float(score),
        raw_bbox=bbox,
        rect_bottom=(float(rect_bottom[0]), float(rect_bottom[1])),
        wide_bottom=(float(wide_bottom[0]), float(wide_bottom[1])),
        wide_bbox=wide_bbox,
    )


def convert_results_to_wide(
    source: str,
    raw_results: List[Tuple[int, float, Tuple[int, int, int, int]]],
    inv_maps: InverseRectifyMaps,
    params: StitchParams,
    right_x_shift: int,
    right_y_shift: int,
    search_radius: int,
    allow_left_overlap: bool,
) -> List[WideDetection]:
    """将某一路的 raw 检测结果列表映射到宽幅坐标。"""
    wide_results = []

    for class_id, score, bbox in raw_results:
        mapped = map_raw_bbox_to_wide(
            source=source,
            bbox=bbox,
            score=score,
            inv_maps=inv_maps,
            params=params,
            right_x_shift=right_x_shift,
            right_y_shift=right_y_shift,
            search_radius=search_radius,
            allow_left_overlap=allow_left_overlap,
        )

        if mapped is not None:
            wide_results.append(mapped)

    return wide_results


# =============================================================================
# 11. 坐标平滑
# =============================================================================

class SmoothTracks:
    """
    简单的检测结果平滑器。

    为什么需要：
        检测框会因为模型输出抖动而轻微跳动。
        自动运镜如果直接使用检测点，裁切框也会抖。
        所以对 wide_bottom 和 wide_bbox 做指数平滑。

    匹配策略：
        每一帧根据 wide_bottom 的欧氏距离，把新检测匹配到上一帧最近的 track。
        如果距离超过 max_match_dist，则认为是新目标。

    smooth 参数含义：
        smooth = 0.0  不平滑，完全使用当前检测
        smooth = 0.65 65% 使用上一帧，35% 使用当前检测
        smooth 越大，画面越稳，但响应越慢
    """

    def __init__(self, smooth: float = 0.65, max_match_dist: float = 180.0, max_missing: int = 15):
        self.smooth = float(np.clip(smooth, 0.0, 0.98))
        self.max_match_dist = float(max_match_dist)
        self.max_missing = int(max_missing)
        self.next_id = 1
        self.tracks: List[Dict] = []

    def update(self, detections: List[WideDetection]) -> List[WideDetection]:
        """
        输入当前帧映射后的检测结果，输出平滑后的检测结果。
        """
        if self.smooth <= 1e-6:
            for det in detections:
                det.track_id = -1
            return detections

        # 先把所有 track 标记为未匹配
        for tr in self.tracks:
            tr["matched"] = False
            tr["missing"] += 1

        # 为了稳定，按 x 坐标排序后依次匹配
        detections_sorted = sorted(detections, key=lambda d: d.wide_bottom[0])
        output = []

        for det in detections_sorted:
            bx, by = det.wide_bottom

            best_track = None
            best_dist = 1e18

            for tr in self.tracks:
                if tr["matched"]:
                    continue

                tx, ty = tr["bottom"]
                dist = ((bx - tx) ** 2 + (by - ty) ** 2) ** 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_track = tr

            if best_track is None or best_dist > self.max_match_dist:
                # 新目标
                track_id = self.next_id
                self.next_id += 1

                best_track = {
                    "id": track_id,
                    "bottom": (bx, by),
                    "bbox": det.wide_bbox,
                    "missing": 0,
                    "matched": True,
                }
                self.tracks.append(best_track)
            else:
                # 匹配到旧目标，做指数平滑
                sx = self.smooth
                old_bx, old_by = best_track["bottom"]
                new_bx = sx * old_bx + (1.0 - sx) * bx
                new_by = sx * old_by + (1.0 - sx) * by

                old_box = best_track["bbox"]
                new_box = tuple(
                    sx * old_box[i] + (1.0 - sx) * det.wide_bbox[i]
                    for i in range(4)
                )

                best_track["bottom"] = (new_bx, new_by)
                best_track["bbox"] = new_box
                best_track["missing"] = 0
                best_track["matched"] = True

            # 把平滑后的坐标写回检测结果
            det.track_id = int(best_track["id"])
            det.wide_bottom = tuple(best_track["bottom"])
            det.wide_bbox = tuple(best_track["bbox"])
            output.append(det)

        # 清理长时间消失的 track
        self.tracks = [
            tr for tr in self.tracks
            if tr["missing"] <= self.max_missing
        ]

        return output


# =============================================================================
# 12. 绘制与输出
# =============================================================================

def draw_wide_detections(
    wide: np.ndarray,
    detections: List[WideDetection],
    fps: float,
    detect_side: str,
    left_age: int,
    right_age: int,
    last_infer_ms: float,
    frame_idx: int,
    params: StitchParams,
) -> np.ndarray:
    """在宽幅图上绘制检测框、底部中心点和状态信息。"""
    vis = wide.copy()

    # 画出虚拟区域分界线，方便你观察坐标是否落在正确的宽幅位置
    left_keep_w = params.left_keep_x2 - params.left_keep_x1
    overlap_w = params.overlap_px

    x_overlap_start = left_keep_w
    x_right_start = left_keep_w + overlap_w

    cv2.line(vis, (x_overlap_start, 0), (x_overlap_start, vis.shape[0] - 1), (0, 255, 255), 2)
    cv2.line(vis, (x_right_start, 0), (x_right_start, vis.shape[0] - 1), (255, 255, 0), 2)

    for det in detections:
        if det.source == "left":
            color = (0, 255, 0)      # 左路：绿色
            src = "L"
        else:
            color = (255, 0, 0)      # 右路：蓝色
            src = "R"

        x1, y1, x2, y2 = det.wide_bbox
        bx, by = det.wide_bottom

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        bc = (int(round(bx)), int(round(by)))

        cv2.rectangle(vis, p1, p2, color, 3)
        cv2.circle(vis, bc, 8, (0, 0, 255), -1)

        label = f"{src} id:{det.track_id} {det.score:.2f} ({bx:.0f},{by:.0f})"
        text_x = max(0, p1[0])
        text_y = max(30, p1[1] - 8)

        cv2.putText(
            vis,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    status = (
        f"FPS:{fps:.1f} persons:{len(detections)} "
        f"detect:{detect_side} infer:{last_infer_ms:.1f}ms "
        f"L_age:{left_age} R_age:{right_age} frame:{frame_idx}"
    )

    cv2.putText(
        vis,
        status,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        vis,
        "yellow line: overlap start | cyan line: right_keep start",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return vis


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    """按比例缩小显示，降低 LCD/窗口显示开销。"""
    if abs(scale - 1.0) < 1e-6:
        return img

    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def print_wide_coords(frame_idx: int, detections: List[WideDetection]) -> None:
    """周期性在终端打印每个人的宽幅坐标。"""
    print(f"[FRAME {frame_idx}] persons={len(detections)}")
    for i, det in enumerate(detections):
        bx, by = det.wide_bottom
        x1, y1, x2, y2 = det.wide_bbox
        print(
            f"  #{i} {det.source} id={det.track_id} conf={det.score:.2f} "
            f"bottom=({bx:.1f},{by:.1f}) "
            f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )


# =============================================================================
# 13. FPS 统计
# =============================================================================

class FPSCounter:
    """简单 FPS 平滑计数器。"""

    def __init__(self):
        self.last_time = None
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 1e-6:
                inst = 1.0 / dt
                self.fps = inst if self.fps <= 1e-6 else 0.9 * self.fps + 0.1 * inst
        self.last_time = now
        return self.fps


# =============================================================================
# 14. 主流程
# =============================================================================

def run(args: argparse.Namespace) -> None:
    """主流程：采集 -> 交替检测 -> 映射 -> 拼接显示。"""
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    ensure_dir(args.save_dir)

    # 1. 加载 map / stitch 参数
    maps_float, maps_fast = load_rectify_maps(args.map_file)
    params = load_stitch_params(args.stitch_param)

    # 2. 构造 raw->rectified 反查表
    inv_maps = load_or_build_inverse_maps(
        maps_float,
        map_file=args.map_file,
        cache_file=args.inverse_cache,
        rebuild=args.rebuild_inverse,
    )

    # 3. 构造左路 left_keep mask，用于屏蔽左路 overlap 区域
    left_keep_mask = build_left_keep_raw_mask(
        maps_float,
        params,
        dilate_iter=args.left_mask_dilate,
    )

    mask_left_overlap = not args.disable_left_keep_mask

    print(f"[信息] 左路 overlap 屏蔽: {mask_left_overlap}")

    # 4. 初始化单个 RKNN 检测器
    #    因为本脚本每次只检测一路，所以一个 RKNNLite 实例即可。
    #    默认 --rknn-core -1 使用全部 NPU core。
    detector = PersonDetector(
        model_path=args.model,
        labels_path=args.labels,
        obj_thresh=args.conf,
        nms_thresh=args.nms,
        input_size=args.input_size,
        core_id=args.rknn_core,
        use_rgb=not args.bgr_input,
        name="alternating-rknn",
    )

    # 5. 打开左右摄像头
    # 使用后台采集线程，避免主循环被 cap.read/grab/retrieve 阻塞。
    cam_left = LatestFrameCamera(
        args.left_device,
        args.width,
        args.height,
        args.fps,
        name="left",
    ).start()

    cam_right = LatestFrameCamera(
        args.right_device,
        args.width,
        args.height,
        args.fps,
        name="right",
    ).start()

    print("[信息] 等待左右摄像头第一帧...")
    if not cam_left.wait_first_frame(timeout=3.0):
        raise RuntimeError("左摄像头 3 秒内没有读到第一帧")
    if not cam_right.wait_first_frame(timeout=3.0):
        raise RuntimeError("右摄像头 3 秒内没有读到第一帧")
    print("[信息] 后台采集线程已启动")

    # 6. 检测结果缓存
    #    注意：每次只检测一路，另一侧使用最近一次结果。
    last_left_results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    last_right_results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []

    last_left_frame_idx = -1
    last_right_frame_idx = -1

    # detect_turn 用于控制左右交替：
    #    0 -> left
    #    1 -> right
    #    2 -> left
    #    3 -> right
    detect_turn = 0

    last_detect_side = "none"
    last_infer_ms = 0.0

    fps_counter = FPSCounter()
    smoother = SmoothTracks(
        smooth=args.smooth,
        max_match_dist=args.smooth_match_dist,
        max_missing=args.smooth_max_missing,
    )

    jsonl_fp = None
    if args.output_jsonl:
        jsonl_fp = open(args.output_jsonl, "w", encoding="utf-8")
        print(f"[信息] 坐标 JSONL 输出: {args.output_jsonl}")

    window_name = "dual RKNN alternating -> wide coords"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    debug_idx = 0

    # last_wide_bg 用于缓存真实宽幅背景。
    # 如果 --wide-refresh-interval > 1，就不是每帧都重新 remap 拼接，
    # 中间帧直接复用上一张宽幅背景，检测框仍然每帧更新。
    last_wide_bg = None

    print("[信息] 开始运行，按 q/ESC 退出，按 s 保存当前宽图 JPG，按 l 切换左路 overlap 屏蔽。")

    try:
        while not STOP_REQUESTED:
            loop_t0 = time.perf_counter()

            # 从后台采集线程取最新帧。
            # 这里不会等待摄像头硬件采集，因此能显著降低主循环 total 时间。
            ok, left_raw, right_raw = get_latest_pair(cam_left, cam_right)
            if not ok:
                print("[警告] 摄像头读取失败")
                time.sleep(0.01)
                continue

            # ------------------------------------------------------------
            # A. 交替检测
            # ------------------------------------------------------------
            do_detect = (frame_idx % max(1, args.detect_interval) == 0)

            if do_detect:
                if detect_turn % 2 == 0:
                    # 检测左路
                    detect_input = left_raw

                    if mask_left_overlap:
                        detect_input = apply_detection_mask(left_raw, left_keep_mask)

                    t0 = time.perf_counter()
                    last_left_results = detector.detect(detect_input)
                    t1 = time.perf_counter()

                    last_left_frame_idx = frame_idx
                    last_detect_side = "left"
                    last_infer_ms = (t1 - t0) * 1000.0

                else:
                    # 检测右路
                    t0 = time.perf_counter()
                    last_right_results = detector.detect(right_raw)
                    t1 = time.perf_counter()

                    last_right_frame_idx = frame_idx
                    last_detect_side = "right"
                    last_infer_ms = (t1 - t0) * 1000.0

                detect_turn += 1

            # ------------------------------------------------------------
            # B. raw 检测结果 -> 宽幅坐标
            # ------------------------------------------------------------
            # 左路默认不允许 overlap，这样符合你的需求：
            #     左路只检测/输出 left_keep；
            #     overlap 和右侧由右路负责。
            left_wide = convert_results_to_wide(
                source="left",
                raw_results=last_left_results,
                inv_maps=inv_maps,
                params=params,
                right_x_shift=args.runtime_right_x_shift,
                right_y_shift=args.runtime_right_y_shift,
                search_radius=args.inverse_search_radius,
                allow_left_overlap=False,
            )

            right_wide = convert_results_to_wide(
                source="right",
                raw_results=last_right_results,
                inv_maps=inv_maps,
                params=params,
                right_x_shift=args.runtime_right_x_shift,
                right_y_shift=args.runtime_right_y_shift,
                search_radius=args.inverse_search_radius,
                allow_left_overlap=True,
            )

            wide_detections = left_wide + right_wide
            wide_detections = smoother.update(wide_detections)

            # ------------------------------------------------------------
            # C. 生成显示背景
            # ------------------------------------------------------------
            # 原始版本每帧都会调用 build_wide_stitch_from_raw()：
            #     left_raw/right_raw -> remap -> 裁剪 -> 融合 -> 完整宽幅图
            #
            # 这一步非常重。你当前日志中：
            #     infer 约 50ms，但 total 约 200ms
            # 说明主要瓶颈不是 RKNN，而是完整宽图 remap/拼接。
            #
            # 因此这里提供三种模式：
            #
            # 1. --display-canvas-only
            #       完全不生成真实宽幅图，只创建黑色坐标画布。
            #       框和底部点仍然画在真实 wide 坐标位置。
            #       这是最快模式，适合测试检测坐标和运镜逻辑。
            #
            # 2. --wide-refresh-interval N
            #       每 N 帧生成一次真实宽幅图，中间帧复用上一张背景。
            #       框每帧更新，背景低频更新。
            #
            # 3. 默认 N=1
            #       每帧生成真实宽幅图，画面最直观，但帧率最低。
            if args.display_canvas_only or args.wide_refresh_interval == 0:
                wide = np.zeros(
                    (int(params.output_height), int(params.output_width), 3),
                    dtype=np.uint8,
                )
            else:
                refresh_n = max(1, int(args.wide_refresh_interval))

                if last_wide_bg is None or frame_idx % refresh_n == 0:
                    last_wide_bg = build_wide_stitch_from_raw(
                        left_raw=left_raw,
                        right_raw=right_raw,
                        maps_fast=maps_fast,
                        params=params,
                        seam_x=args.runtime_seam_x,
                        blend_width=args.runtime_blend_width,
                        right_x_shift=args.runtime_right_x_shift,
                        right_y_shift=args.runtime_right_y_shift,
                    )

                wide = last_wide_bg

            fps = fps_counter.update()

            left_age = frame_idx - last_left_frame_idx if last_left_frame_idx >= 0 else -1
            right_age = frame_idx - last_right_frame_idx if last_right_frame_idx >= 0 else -1

            vis = draw_wide_detections(
                wide=wide,
                detections=wide_detections,
                fps=fps,
                detect_side=last_detect_side,
                left_age=left_age,
                right_age=right_age,
                last_infer_ms=last_infer_ms,
                frame_idx=frame_idx,
                params=params,
            )

            # ------------------------------------------------------------
            # D. 可选输出 JSONL
            # ------------------------------------------------------------
            if jsonl_fp is not None:
                item = {
                    "frame": frame_idx,
                    "detect_side": last_detect_side,
                    "infer_ms": last_infer_ms,
                    "left_age": left_age,
                    "right_age": right_age,
                    "persons": [
                        {
                            "source": det.source,
                            "track_id": det.track_id,
                            "score": det.score,
                            "raw_bbox": list(det.raw_bbox),
                            "rect_bottom": list(det.rect_bottom),
                            "wide_bottom": list(det.wide_bottom),
                            "wide_bbox": list(det.wide_bbox),
                        }
                        for det in wide_detections
                    ],
                }
                jsonl_fp.write(json.dumps(item, ensure_ascii=False) + "\n")

            # ------------------------------------------------------------
            # E. 显示和按键
            # ------------------------------------------------------------
            if not args.headless:
                show = resize_for_display(vis, args.display_scale)
                cv2.imshow(window_name, show)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            term_key = read_terminal_key()
            if term_key != 255:
                key = term_key

            if key in (ord("q"), 27):
                print("[信息] 用户退出")
                break

            elif key == ord("s"):
                path = os.path.join(
                    args.save_dir,
                    f"wide_debug_{debug_idx:04d}_{current_timestamp()}.jpg",
                )
                cv2.imwrite(path, vis)
                print(f"[信息] 已保存宽图调试 JPG: {path}")
                debug_idx += 1

            elif key == ord("l"):
                mask_left_overlap = not mask_left_overlap
                print(f"[信息] 左路 overlap 屏蔽: {mask_left_overlap}")

            # ------------------------------------------------------------
            # F. 周期性终端输出
            # ------------------------------------------------------------
            if args.print_every > 0 and frame_idx % args.print_every == 0:
                loop_t1 = time.perf_counter()
                total_ms = (loop_t1 - loop_t0) * 1000.0
                print(
                    f"[PROFILE] frame={frame_idx} fps={fps:.1f} "
                    f"detect={last_detect_side} infer={last_infer_ms:.1f}ms "
                    f"total={total_ms:.1f}ms "
                    f"L={len(last_left_results)} age={left_age} "
                    f"R={len(last_right_results)} age={right_age} "
                    f"wide_persons={len(wide_detections)}"
                )
                if args.print_coords:
                    print_wide_coords(frame_idx, wide_detections)

            frame_idx += 1

    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

        try:
            cam_left.stop()
            cam_right.stop()
        except Exception:
            pass

        try:
            detector.close()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("[信息] 程序退出")


# =============================================================================
# 15. 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual camera RKNN alternating detection -> wide coordinate display."
    )

    # 摄像头
    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")
    parser.add_argument("--width", type=int, default=1920, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=30, help="摄像头采集帧率")

    # RKNN 模型
    parser.add_argument("--model", default=DEFAULT_MODEL, help="RKNN 模型路径，建议使用已验证的 best_2.rknn")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="labels.txt 路径")
    parser.add_argument("--input-size", type=int, default=MODEL_INPUT_SIZE_DEFAULT, help="模型输入尺寸，默认 640")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument(
        "--rknn-core",
        type=int,
        default=-1,
        help="RKNN NPU core：0/1/2 指定单核，-1 使用 NPU_CORE_0_1_2",
    )
    parser.add_argument("--bgr-input", action="store_true", help="模型输入为 BGR；默认会 BGR->RGB")

    # map / stitch
    parser.add_argument("--map-file", default=DEFAULT_MAP_FILE, help="stereo_rectify_maps_wide_good.npz")
    parser.add_argument("--stitch-param", default=DEFAULT_STITCH_PARAM, help="stitch_params_good.npz")
    parser.add_argument("--inverse-cache", default=None, help="raw->rectified 反查表缓存路径，默认自动生成")
    parser.add_argument("--rebuild-inverse", action="store_true", help="强制重建 raw->rectified 反查表")
    parser.add_argument("--inverse-search-radius", type=int, default=8, help="raw 点反查 rectified 时的邻域搜索半径")

    # 拼接微调参数：默认使用你目前调好的参数
    parser.add_argument("--runtime-seam-x", type=int, default=150, help="overlap 内接缝中心位置")
    parser.add_argument("--runtime-blend-width", type=int, default=40, help="接缝融合宽度")
    parser.add_argument("--runtime-right-x-shift", type=int, default=30, help="右图 rectified x 微调")
    parser.add_argument("--runtime-right-y-shift", type=int, default=-5, help="右图 rectified y 微调")

    # 交替检测参数
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=1,
        help=(
            "检测间隔。1 表示每帧检测一路，左右交替；"
            "2 表示每 2 帧检测一路；值越大，显示越流畅但检测更新更慢。"
        ),
    )

    # 左路 overlap 屏蔽
    parser.add_argument(
        "--disable-left-keep-mask",
        action="store_true",
        help="关闭左路 left_keep mask；默认左路 overlap 会被屏蔽",
    )
    parser.add_argument(
        "--left-mask-dilate",
        type=int,
        default=3,
        help="left_keep raw mask 膨胀次数，适当填补 mask 小孔洞",
    )

    # 平滑
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.65,
        help="坐标平滑系数，0 表示不平滑，0.65 表示较稳，越大响应越慢",
    )
    parser.add_argument("--smooth-match-dist", type=float, default=180.0, help="平滑跟踪匹配最大距离")
    parser.add_argument("--smooth-max-missing", type=int, default=15, help="track 最大丢失帧数")

    # 显示与调试
    parser.add_argument("--display-scale", type=float, default=0.25, help="显示缩放比例")
    parser.add_argument(
        "--display-canvas-only",
        action="store_true",
        help="只显示黑色宽幅坐标画布，不生成真实宽幅图。速度最快，适合测试检测坐标和运镜逻辑",
    )
    parser.add_argument(
        "--wide-refresh-interval",
        type=int,
        default=1,
        help="真实宽幅背景刷新间隔。1=每帧刷新；5=每5帧刷新一次；0=不刷新，等价于 canvas-only",
    )
    parser.add_argument("--headless", action="store_true", help="不显示窗口，只打印坐标")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="按 s 保存宽图 JPG 的目录")
    parser.add_argument("--print-every", type=int, default=10, help="每隔 N 帧打印 profile，0 表示不打印")
    parser.add_argument("--print-coords", action="store_true", help="打印每个人的宽幅坐标")
    parser.add_argument("--output-jsonl", default=None, help="可选：保存每帧宽幅坐标 JSONL，不保存视频")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ 双路 RKNN 交替检测 + 宽幅坐标显示 ================")
    print(f"left_device           : {args.left_device}")
    print(f"right_device          : {args.right_device}")
    print(f"camera size           : {args.width} x {args.height}")
    print(f"camera fps            : {args.fps}")
    print(f"model                 : {args.model}")
    print(f"labels                : {args.labels}")
    print(f"map_file              : {args.map_file}")
    print(f"stitch_param          : {args.stitch_param}")
    print(f"conf / nms            : {args.conf} / {args.nms}")
    print(f"rknn_core             : {args.rknn_core}")
    print(f"detect_interval       : {args.detect_interval}")
    print(f"smooth                : {args.smooth}")
    print(f"runtime seam/blend    : {args.runtime_seam_x} / {args.runtime_blend_width}")
    print(f"runtime right shift   : x={args.runtime_right_x_shift}, y={args.runtime_right_y_shift}")
    print("=================================================================\n")

    run(args)


if __name__ == "__main__":
    main()
