#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_select_h264_rkmpp_video.py

RK3588 / ELF2 多路视频录制脚本。

功能：
    1. 双路 USB 摄像头 1920x1080 采集。
    2. 可以按键分别录制：
        - 左单目视频 left_mono（可选 raw 或 rectified，默认 rectified）
        - 右单目视频 right_mono（可选 raw 或 rectified，默认 rectified）
        - 完整宽幅拼接视频 wide
        - 1920x1080 裁切视频 view1920
    3. 使用 FFmpeg + h264_rkmpp 硬件编码，而不是 OpenCV VideoWriter。
    4. 每一路视频都有独立写入线程和队列，避免编码写盘直接阻塞主循环。

按键：
    1 : 开始/停止录制 left_mono
    2 : 开始/停止录制 right_mono
    3 : 开始/停止录制 wide 宽幅
    4 : 开始/停止录制 view1920
    a : 全部开始 / 全部停止
    s : 保存当前帧 JPG 调试图
    h : 显示帮助
    q / Esc : 退出

典型运行：
    python3 record_select_h264_rkmpp_video.py \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
        --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
        --save-root /home/elf/work/basketball/h264_recordings \
        --width 1920 --height 1080 --fps 30 \
        --view-fps 20 --wide-fps 5 --left-fps 20 --right-fps 20 \
        --runtime-seam-x 150 \
        --runtime-blend-width 40 \
        --runtime-right-x-shift 30 \
        --runtime-right-y-shift -5 \
        --display-scale 0.25

重要建议：
    - view1920 是最终主输出，建议 20 FPS 或更高。
    - wide 分辨率约 3406x1202，数据量很大，建议 5~10 FPS 用于调试。
    - 如果同时录 4 路，仍然会有内存带宽和写盘压力，必要时降低 wide-fps。
    - left/right 单目默认保存“极线矫正后的 rectified 图”，如果想保存摄像头原图，启动时加 --mono-source raw。
"""

import argparse
import os
import sys
import time
import select
import queue
import shutil
import threading
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

# RK3588 / Mali 平台上 OpenCV 有时会尝试启用 OpenCL，可能带来额外开销。
# 必须在 import cv2 前设置。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# ============================================================
# 1. 默认参数
# ============================================================

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"
DEFAULT_MAP_FILE = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz"
DEFAULT_STITCH_PARAM = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_SAVE_ROOT = "/home/elf/work/basketball/h264_recordings"


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class RectifyMaps:
    """raw -> rectified 的 remap 查找表。"""
    raw_image_size: Tuple[int, int]
    rectified_size: Tuple[int, int]
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


@dataclass
class StitchParams:
    """离线拼接参数。"""
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


# ============================================================
# 3. 工具函数
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 格式不正确: {value}")
    return int(flat[0]), int(flat[1])


def get_npz_int(data: np.lib.npyio.NpzFile, key: str, default: Optional[int] = None) -> int:
    if key not in data.files:
        if default is None:
            raise RuntimeError(f"npz 缺少字段: {key}")
        return int(default)
    return int(np.array(data[key]).reshape(-1)[0])


def convert_maps_for_fast_remap(map1: np.ndarray, map2: np.ndarray):
    """把 float32 remap map 转成 fixed-point map，让 cv2.remap 更快。"""
    if map1.dtype == np.int16 and map2.dtype == np.uint16:
        return map1, map2
    return cv2.convertMaps(map1.astype(np.float32), map2.astype(np.float32), cv2.CV_16SC2)


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def check_dir_writable(path: str) -> None:
    ensure_dir(path)
    test_file = os.path.join(path, ".write_test.tmp")
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise RuntimeError(f"目录不可写: {path}\n原因: {e}")


def read_terminal_key() -> int:
    """终端非阻塞读取按键：输入 1/2/3/4/a/s/q 后回车也能控制。"""
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


def make_even_for_video(img: np.ndarray) -> np.ndarray:
    """H.264 编码通常要求宽高为偶数，奇数时在右侧/底部补黑边。"""
    h, w = img.shape[:2]
    pad_right = 1 if (w % 2) else 0
    pad_bottom = 1 if (h % 2) else 0
    if pad_right == 0 and pad_bottom == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def crop_view_from_wide(wide: np.ndarray, crop_x: int, crop_y: int, view_w: int, view_h: int) -> np.ndarray:
    """从完整宽幅图中裁出 1920x1080；crop=-1 表示居中。"""
    h, w = wide.shape[:2]
    if crop_x < 0:
        crop_x = (w - view_w) // 2
    if crop_y < 0:
        crop_y = (h - view_h) // 2
    crop_x = int(np.clip(crop_x, 0, max(0, w - view_w)))
    crop_y = int(np.clip(crop_y, 0, max(0, h - view_h)))
    view = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    valid_w = min(view_w, w - crop_x)
    valid_h = min(view_h, h - crop_y)
    if valid_w > 0 and valid_h > 0:
        view[:valid_h, :valid_w] = wide[crop_y:crop_y + valid_h, crop_x:crop_x + valid_w]
    return view


# ============================================================
# 4. 加载离线文件
# ============================================================

def load_rectify_maps(map_file: str) -> RectifyMaps:
    if not os.path.exists(map_file):
        raise RuntimeError(f"找不到 map 文件: {map_file}")
    data = np.load(map_file)
    required = ["left_rect_map1", "left_rect_map2", "right_rect_map1", "right_rect_map2", "raw_image_size"]
    for key in required:
        if key not in data.files:
            raise RuntimeError(f"map 文件缺少字段: {key}")

    raw_image_size = parse_image_size(data["raw_image_size"])
    if "rectified_size" in data.files:
        rectified_size = parse_image_size(data["rectified_size"])
    else:
        rectified_size = (int(data["left_rect_map1"].shape[1]), int(data["left_rect_map1"].shape[0]))

    left_map1, left_map2 = convert_maps_for_fast_remap(data["left_rect_map1"], data["left_rect_map2"])
    right_map1, right_map2 = convert_maps_for_fast_remap(data["right_rect_map1"], data["right_rect_map2"])

    maps = RectifyMaps(raw_image_size, rectified_size, left_map1, left_map2, right_map1, right_map2)
    print("已加载 remap 文件:")
    print(f"  map_file       : {map_file}")
    print(f"  raw_image_size : {maps.raw_image_size}")
    print(f"  rectified_size : {maps.rectified_size}")
    return maps


def load_stitch_params(stitch_param_file: str) -> StitchParams:
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
    print("\n已加载拼接参数:")
    print(f"  stitch_param    : {stitch_param_file}")
    print(f"  overlap_px      : {params.overlap_px}")
    print(f"  vertical_offset : {params.vertical_offset}")
    print(f"  blend_width     : {params.blend_width}")
    print(f"  output_size     : {params.output_width} x {params.output_height}")
    print(f"  left_y          : {params.left_y1} -> {params.left_y2}")
    print(f"  right_y         : {params.right_y1} -> {params.right_y2}")
    return params


# ============================================================
# 5. 摄像头采集线程
# ============================================================

def open_usb_camera(device: str, width: int, height: int, fps: int, use_mjpg: bool = True) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))

    print(f"\n摄像头已打开: {device}")
    print(f"  request size : {width}x{height}")
    print(f"  real size    : {real_w}x{real_h}")
    print(f"  request fps  : {fps}")
    print(f"  real fps     : {real_fps}")
    print(f"  fourcc       : {fourcc_str}")
    return cap


class LatestFrameCamera:
    """后台采集线程：cap.read() 不阻塞主循环，主线程只取最新帧。"""
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
            if not ret or frame is None:
                with self.lock:
                    self.ok = False
                time.sleep(0.002)
                continue
            with self.lock:
                self.frame = frame
                self.ok = True

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None
            return True, self.frame

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass


def grab_pair_latest(left_cam: LatestFrameCamera, right_cam: LatestFrameCamera):
    ok_l, left = left_cam.read_latest()
    ok_r, right = right_cam.read_latest()
    if not ok_l or not ok_r or left is None or right is None:
        return False, None, None
    return True, left, right


# ============================================================
# 6. remap 与拼接
# ============================================================

def remap_rectified_roi_fixed_size(raw, map1, map2, rect_x1, rect_y1, rect_x2, rect_y2) -> np.ndarray:
    """remap 出 rectified 坐标下 ROI；越界区域用黑边补齐，保证输出尺寸固定。"""
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

    roi = cv2.remap(raw, map1[cy1:cy2, cx1:cx2], map2[cy1:cy2, cx1:cx2],
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dx1 = cx1 - int(rect_x1)
    dy1 = cy1 - int(rect_y1)
    out[dy1:dy1 + roi.shape[0], dx1:dx1 + roi.shape[1]] = roi
    return out


def rectify_full_frame(raw: np.ndarray, map1: np.ndarray, map2: np.ndarray, maps: RectifyMaps) -> np.ndarray:
    """
    生成完整的极线矫正单目图。

    用途：
        录制 left/right 单目视频时，如果 --mono-source rectified，
        保存的不是摄像头 raw 图，而是 raw -> rectified 之后的图。

    输出尺寸：
        maps.rectified_size，例如你的 wide map 是 2208 x 1242。

    注意：
        这个是完整 remap，计算量比直接保存 raw 大。
        所以只有在预览/录制/调试需要 left/right 单目时才会计算。
    """
    if (raw.shape[1], raw.shape[0]) != maps.raw_image_size:
        raw = cv2.resize(raw, maps.raw_image_size)

    return cv2.remap(
        raw,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def build_wide_stitch_from_raw(left_raw, right_raw, maps, params, seam_x, blend_width, right_x_shift, right_y_shift):
    """生成完整宽幅图。宽幅分辨率大，建议只低 FPS 录制。"""
    if (left_raw.shape[1], left_raw.shape[0]) != maps.raw_image_size:
        left_raw = cv2.resize(left_raw, maps.raw_image_size)
    if (right_raw.shape[1], right_raw.shape[0]) != maps.raw_image_size:
        right_raw = cv2.resize(right_raw, maps.raw_image_size)

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

    left_keep = remap_rectified_roi_fixed_size(left_raw, maps.left_map1, maps.left_map2,
                                               params.left_keep_x1, left_y1, params.left_keep_x2, left_y2)
    wide[:, left_keep_start:left_keep_end] = left_keep

    blend_width = max(1, min(int(blend_width), overlap_w))
    seam_x_used = overlap_w // 2 if seam_x < 0 else int(seam_x)
    half_blend = blend_width // 2
    seam_x_used = int(np.clip(seam_x_used, half_blend, overlap_w - (blend_width - half_blend)))
    blend_x1 = seam_x_used - half_blend
    blend_x2 = blend_x1 + blend_width

    if blend_x1 > 0:
        left_overlap_left = remap_rectified_roi_fixed_size(left_raw, maps.left_map1, maps.left_map2,
                                                           params.left_overlap_x1, left_y1,
                                                           params.left_overlap_x1 + blend_x1, left_y2)
        wide[:, overlap_start:overlap_start + blend_x1] = left_overlap_left

    if blend_x2 < overlap_w:
        right_overlap_right = remap_rectified_roi_fixed_size(right_raw, maps.right_map1, maps.right_map2,
                                                             params.right_overlap_x1 + blend_x2 + right_x_shift,
                                                             right_y1,
                                                             params.right_overlap_x1 + overlap_w + right_x_shift,
                                                             right_y2)
        wide[:, overlap_start + blend_x2:overlap_end] = right_overlap_right

    left_blend = remap_rectified_roi_fixed_size(left_raw, maps.left_map1, maps.left_map2,
                                                params.left_overlap_x1 + blend_x1, left_y1,
                                                params.left_overlap_x1 + blend_x2, left_y2)
    right_blend = remap_rectified_roi_fixed_size(right_raw, maps.right_map1, maps.right_map2,
                                                 params.right_overlap_x1 + blend_x1 + right_x_shift,
                                                 right_y1,
                                                 params.right_overlap_x1 + blend_x2 + right_x_shift,
                                                 right_y2)
    alpha = np.linspace(0.0, 1.0, blend_width, dtype=np.float32).reshape(1, blend_width, 1)
    blend = left_blend.astype(np.float32) * (1.0 - alpha) + right_blend.astype(np.float32) * alpha
    wide[:, overlap_start + blend_x1:overlap_start + blend_x2] = np.clip(blend, 0, 255).astype(np.uint8)

    right_keep = remap_rectified_roi_fixed_size(right_raw, maps.right_map1, maps.right_map2,
                                                params.right_keep_x1 + right_x_shift, right_y1,
                                                params.right_keep_x2 + right_x_shift, right_y2)
    wide[:, right_keep_start:right_keep_end] = right_keep
    return wide


def build_view_stitch_from_raw(left_raw, right_raw, maps, params, crop_x, crop_y, view_w, view_h,
                               seam_x, blend_width, right_x_shift, right_y_shift):
    """直接生成 view1920，不生成完整宽幅图，速度比先生成 wide 再裁切更快。"""
    if (left_raw.shape[1], left_raw.shape[0]) != maps.raw_image_size:
        left_raw = cv2.resize(left_raw, maps.raw_image_size)
    if (right_raw.shape[1], right_raw.shape[0]) != maps.raw_image_size:
        right_raw = cv2.resize(right_raw, maps.raw_image_size)

    output_w = int(params.output_width)
    output_h = int(params.output_height)
    if crop_x < 0:
        crop_x = (output_w - view_w) // 2
    if crop_y < 0:
        crop_y = (output_h - view_h) // 2
    crop_x = int(np.clip(crop_x, 0, max(0, output_w - view_w)))
    crop_y = int(np.clip(crop_y, 0, max(0, output_h - view_h)))

    view = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    valid_h = min(view_h, output_h - crop_y)
    if valid_h <= 0:
        return view

    left_y1 = params.left_y1 + crop_y
    left_y2 = left_y1 + valid_h
    right_y1 = params.right_y1 + crop_y + right_y_shift
    right_y2 = right_y1 + valid_h

    left_keep_w = params.left_keep_x2 - params.left_keep_x1
    overlap_w = params.overlap_px
    right_keep_w = params.right_keep_x2 - params.right_keep_x1
    left_keep_start = 0
    left_keep_end = left_keep_w
    overlap_start = left_keep_end
    overlap_end = overlap_start + overlap_w
    right_keep_start = overlap_end
    right_keep_end = right_keep_start + right_keep_w
    view_start = crop_x
    view_end = crop_x + view_w

    def paste_direct(seg_start, seg_end, raw, map1, map2, rect_x_base, rect_y1, rect_y2):
        ix1 = max(view_start, seg_start)
        ix2 = min(view_end, seg_end)
        if ix2 <= ix1:
            return
        dst_x1 = ix1 - view_start
        dst_x2 = ix2 - view_start
        rect_x1 = rect_x_base + (ix1 - seg_start)
        rect_x2 = rect_x_base + (ix2 - seg_start)
        roi = remap_rectified_roi_fixed_size(raw, map1, map2, rect_x1, rect_y1, rect_x2, rect_y2)
        view[:valid_h, dst_x1:dst_x2] = roi

    paste_direct(left_keep_start, left_keep_end, left_raw, maps.left_map1, maps.left_map2,
                 params.left_keep_x1, left_y1, left_y2)

    ix1 = max(view_start, overlap_start)
    ix2 = min(view_end, overlap_end)
    if ix2 > ix1:
        ox1 = ix1 - overlap_start
        ox2 = ix2 - overlap_start
        blend_width = max(1, min(int(blend_width), overlap_w))
        seam_x_used = overlap_w // 2 if seam_x < 0 else int(seam_x)
        half_blend = blend_width // 2
        seam_x_used = int(np.clip(seam_x_used, half_blend, overlap_w - (blend_width - half_blend)))
        blend_x1 = seam_x_used - half_blend
        blend_x2 = blend_x1 + blend_width

        def paste_overlap_left(local_x1, local_x2):
            if local_x2 <= local_x1:
                return
            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start
            roi = remap_rectified_roi_fixed_size(left_raw, maps.left_map1, maps.left_map2,
                                                 params.left_overlap_x1 + local_x1, left_y1,
                                                 params.left_overlap_x1 + local_x2, left_y2)
            view[:valid_h, dst_x1:dst_x2] = roi

        def paste_overlap_right(local_x1, local_x2):
            if local_x2 <= local_x1:
                return
            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start
            roi = remap_rectified_roi_fixed_size(right_raw, maps.right_map1, maps.right_map2,
                                                 params.right_overlap_x1 + local_x1 + right_x_shift,
                                                 right_y1,
                                                 params.right_overlap_x1 + local_x2 + right_x_shift,
                                                 right_y2)
            view[:valid_h, dst_x1:dst_x2] = roi

        paste_overlap_left(ox1, min(ox2, blend_x1))
        paste_overlap_right(max(ox1, blend_x2), ox2)

        bx1 = max(ox1, blend_x1)
        bx2 = min(ox2, blend_x2)
        if bx2 > bx1:
            dst_x1 = overlap_start + bx1 - view_start
            dst_x2 = overlap_start + bx2 - view_start
            left_roi = remap_rectified_roi_fixed_size(left_raw, maps.left_map1, maps.left_map2,
                                                      params.left_overlap_x1 + bx1, left_y1,
                                                      params.left_overlap_x1 + bx2, left_y2)
            right_roi = remap_rectified_roi_fixed_size(right_raw, maps.right_map1, maps.right_map2,
                                                       params.right_overlap_x1 + bx1 + right_x_shift,
                                                       right_y1,
                                                       params.right_overlap_x1 + bx2 + right_x_shift,
                                                       right_y2)
            alpha_line = np.linspace(0.0, 1.0, blend_width, dtype=np.float32)
            alpha = alpha_line[bx1 - blend_x1:bx2 - blend_x1].reshape(1, -1, 1)
            blended = left_roi.astype(np.float32) * (1.0 - alpha) + right_roi.astype(np.float32) * alpha
            view[:valid_h, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    paste_direct(right_keep_start, right_keep_end, right_raw, maps.right_map1, maps.right_map2,
                 params.right_keep_x1 + right_x_shift, right_y1, right_y2)
    return view


# ============================================================
# 7. FFmpeg 硬件编码异步录制器
# ============================================================

class AsyncFFmpegRecorder:
    """单路异步 FFmpeg 录制器：主线程投递帧，后台线程写入 FFmpeg stdin。"""
    def __init__(self, name, out_dir, prefix, fps, bitrate, encoder, container,
                 ffmpeg_path, queue_size, gop, loglevel, drop_policy="oldest"):
        self.name = name
        self.out_dir = out_dir
        self.prefix = prefix
        self.fps = float(fps)
        self.bitrate = bitrate
        self.encoder = encoder
        self.container = container.lstrip(".")
        self.ffmpeg_path = ffmpeg_path
        self.queue_size = int(queue_size)
        self.gop = int(gop)
        self.loglevel = loglevel
        self.drop_policy = drop_policy
        self.proc = None
        self.thread = None
        self.q = None
        self.width = None
        self.height = None
        self.path = None
        self.active = False
        self.stop_requested = False
        self.written = 0
        self.dropped = 0
        self.enqueued = 0
        self.start_time = None
        self.next_frame_time = 0.0

    def start(self, first_frame: np.ndarray) -> None:
        if self.active:
            return
        ensure_dir(self.out_dir)
        frame = make_even_for_video(first_frame)
        h, w = frame.shape[:2]
        self.width = int(w)
        self.height = int(h)
        ts = current_timestamp()
        self.path = os.path.join(self.out_dir, f"{self.prefix}_{ts}.{self.container}")

        cmd = [
            self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", self.loglevel,
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{self.width}x{self.height}",
            "-r", f"{self.fps}", "-i", "pipe:0",
            "-an", "-vf", "format=nv12",
            "-c:v", self.encoder,
            "-b:v", self.bitrate,
            "-g", str(self.gop),
            self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=None, bufsize=0)
        self.q = queue.Queue(maxsize=self.queue_size)
        self.stop_requested = False
        self.active = True
        self.written = 0
        self.dropped = 0
        self.enqueued = 0
        self.start_time = time.time()
        self.next_frame_time = 0.0
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        print(f"\n[{self.name}] 开始录制:")
        print(f"  path    : {self.path}")
        print(f"  size    : {self.width} x {self.height}")
        print(f"  fps     : {self.fps}")
        print(f"  encoder : {self.encoder}")
        print(f"  bitrate : {self.bitrate}")

    def should_take_frame(self, now: float) -> bool:
        if not self.active:
            return False
        if self.next_frame_time <= 0.0:
            self.next_frame_time = now
            return True
        if now + 1e-6 < self.next_frame_time:
            return False
        period = 1.0 / max(self.fps, 1e-6)
        while self.next_frame_time <= now:
            self.next_frame_time += period
        return True

    def enqueue(self, frame: np.ndarray, now: float) -> None:
        if not self.active or not self.should_take_frame(now):
            return
        frame = make_even_for_video(frame)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame)
        try:
            self.q.put_nowait(frame)
            self.enqueued += 1
        except queue.Full:
            self.dropped += 1
            if self.drop_policy == "oldest":
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(frame)
                    self.enqueued += 1
                except queue.Full:
                    self.dropped += 1

    def _writer_loop(self) -> None:
        assert self.proc is not None and self.proc.stdin is not None and self.q is not None
        while True:
            if self.stop_requested and self.q.empty():
                break
            try:
                frame = self.q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.proc.stdin.write(frame.tobytes())
                self.written += 1
            except BrokenPipeError:
                print(f"[{self.name}] FFmpeg stdin 已断开，停止写入。")
                break
            except Exception as e:
                print(f"[{self.name}] 写入 FFmpeg 失败: {e}")
                break
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print(f"[{self.name}] FFmpeg 退出超时，强制终止。")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def stop(self) -> None:
        if not self.active:
            return
        print(f"\n[{self.name}] 正在停止录制，请稍等...")
        self.stop_requested = True
        if self.thread is not None:
            self.thread.join(timeout=10.0)
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        print(f"[{self.name}] 已停止:")
        print(f"  path    : {self.path}")
        print(f"  written : {self.written}")
        print(f"  dropped : {self.dropped}")
        print(f"  elapsed : {elapsed:.1f} s")
        if elapsed > 0:
            print(f"  actual write fps approx: {self.written / elapsed:.1f}")
        self.proc = None
        self.thread = None
        self.q = None
        self.active = False
        self.stop_requested = False

    def queue_state(self) -> str:
        qsize = self.q.qsize() if self.q is not None else 0
        return f"{self.name}:on={self.active},q={qsize},w={self.written},d={self.dropped}"


# ============================================================
# 8. 主流程
# ============================================================

def print_key_help() -> None:
    print("\n按键：")
    print("  1       : 开始/停止录制 left_mono（raw 或 rectified，由 --mono-source 决定）")
    print("  2       : 开始/停止录制 right_mono（raw 或 rectified，由 --mono-source 决定）")
    print("  3       : 开始/停止录制 wide 宽幅")
    print("  4       : 开始/停止录制 view1920")
    print("  a       : 全部开始 / 全部停止")
    print("  s       : 保存当前帧 JPG 调试图")
    print("  h       : 显示帮助")
    print("  q / Esc : 退出\n")


def start_or_stop_recorder(rec: AsyncFFmpegRecorder, frame: np.ndarray) -> None:
    if rec.active:
        rec.stop()
    else:
        rec.start(frame)


def run(args: argparse.Namespace) -> None:
    ffmpeg_path = shutil.which(args.ffmpeg)
    if ffmpeg_path is None:
        raise RuntimeError(f"找不到 ffmpeg: {args.ffmpeg}")

    maps = load_rectify_maps(args.map_file)
    params = load_stitch_params(args.stitch_param)

    # left/right 单目保存目录根据 --mono-source 自动命名。
    #   rectified: 保存极线矫正后的图，目录默认 left_rectified/right_rectified
    #   raw      : 保存摄像头原始图，目录默认 left_raw/right_raw
    mono_left_name = "left_rectified" if args.mono_source == "rectified" else "left_raw"
    mono_right_name = "right_rectified" if args.mono_source == "rectified" else "right_raw"

    left_dir = args.left_dir or os.path.join(args.save_root, mono_left_name)
    right_dir = args.right_dir or os.path.join(args.save_root, mono_right_name)
    wide_dir = args.wide_dir or os.path.join(args.save_root, "wide")
    view_dir = args.view_dir or os.path.join(args.save_root, "view1920")
    debug_dir = args.debug_dir or os.path.join(args.save_root, "debug_jpg")
    for d in [left_dir, right_dir, wide_dir, view_dir, debug_dir]:
        check_dir_writable(d)

    print("\n运行时参数:")
    print(f"  encoder              : {args.encoder}")
    print(f"  container            : {args.container}")
    print(f"  ffmpeg               : {ffmpeg_path}")
    print(f"  runtime_seam_x       : {args.runtime_seam_x}")
    print(f"  runtime_blend_width  : {args.runtime_blend_width}")
    print(f"  runtime_right_x_shift: {args.runtime_right_x_shift}")
    print(f"  runtime_right_y_shift: {args.runtime_right_y_shift}")
    print(f"  view size            : {args.view_width} x {args.view_height}")
    print(f"  mono_source          : {args.mono_source}")
    print(f"  left/right fps       : {args.left_fps} / {args.right_fps}")
    print(f"  wide/view fps        : {args.wide_fps} / {args.view_fps}")
    print(f"  save_root            : {args.save_root}")
    print(f"  left_dir             : {left_dir}")
    print(f"  right_dir            : {right_dir}")

    cap_left = open_usb_camera(args.left_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)
    cap_right = open_usb_camera(args.right_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)

    left_prefix = "left_rectified" if args.mono_source == "rectified" else "left_raw"
    right_prefix = "right_rectified" if args.mono_source == "rectified" else "right_raw"

    rec_left = AsyncFFmpegRecorder("left", left_dir, left_prefix, args.left_fps, args.left_bitrate, args.encoder, args.container, ffmpeg_path, args.queue_size, args.gop, args.ffmpeg_loglevel)
    rec_right = AsyncFFmpegRecorder("right", right_dir, right_prefix, args.right_fps, args.right_bitrate, args.encoder, args.container, ffmpeg_path, args.queue_size, args.gop, args.ffmpeg_loglevel)
    rec_wide = AsyncFFmpegRecorder("wide", wide_dir, "wide", args.wide_fps, args.wide_bitrate, args.encoder, args.container, ffmpeg_path, args.queue_size, args.gop, args.ffmpeg_loglevel)
    rec_view = AsyncFFmpegRecorder("view", view_dir, "view1920", args.view_fps, args.view_bitrate, args.encoder, args.container, ffmpeg_path, args.queue_size, args.gop, args.ffmpeg_loglevel)
    recorders = {"left": rec_left, "right": rec_right, "wide": rec_wide, "view": rec_view}

    left_cam = None
    right_cam = None
    stat_count = 0
    stat_total_ms = 0.0
    stat_build_ms = 0.0
    debug_idx = 0

    try:
        left_cam = LatestFrameCamera(cap_left, "left").start()
        right_cam = LatestFrameCamera(cap_right, "right").start()

        print("\n等待摄像头第一帧...")
        for _ in range(200):
            ok, _, _ = grab_pair_latest(left_cam, right_cam)
            if ok:
                break
            time.sleep(0.01)

        print_key_help()

        while True:
            loop_t0 = time.perf_counter()
            ok, left_raw, right_raw = grab_pair_latest(left_cam, right_cam)
            if not ok:
                print("[警告] 摄像头读取失败")
                time.sleep(0.01)
                continue

            wide = None
            view = None
            build_ms = 0.0

            def ensure_wide_and_view():
                nonlocal wide, view, build_ms
                if wide is None:
                    t0 = time.perf_counter()
                    wide = build_wide_stitch_from_raw(left_raw, right_raw, maps, params,
                                                      args.runtime_seam_x, args.runtime_blend_width,
                                                      args.runtime_right_x_shift, args.runtime_right_y_shift)
                    build_ms += (time.perf_counter() - t0) * 1000.0
                if view is None:
                    view = crop_view_from_wide(wide, args.crop_x, args.crop_y, args.view_width, args.view_height)
                return wide, view

            def ensure_view_only():
                nonlocal view, build_ms
                if view is None:
                    t0 = time.perf_counter()
                    view = build_view_stitch_from_raw(left_raw, right_raw, maps, params,
                                                      args.crop_x, args.crop_y,
                                                      args.view_width, args.view_height,
                                                      args.runtime_seam_x, args.runtime_blend_width,
                                                      args.runtime_right_x_shift, args.runtime_right_y_shift)
                    build_ms += (time.perf_counter() - t0) * 1000.0
                return view

            left_mono = None
            right_mono = None

            def ensure_left_mono():
                nonlocal left_mono, build_ms
                if left_mono is not None:
                    return left_mono
                if args.mono_source == "raw":
                    left_mono = left_raw
                else:
                    # 保存拼接前的左单目：raw -> rectified 后的完整极线矫正图。
                    t0 = time.perf_counter()
                    left_mono = rectify_full_frame(left_raw, maps.left_map1, maps.left_map2, maps)
                    build_ms += (time.perf_counter() - t0) * 1000.0
                return left_mono

            def ensure_right_mono():
                nonlocal right_mono, build_ms
                if right_mono is not None:
                    return right_mono
                if args.mono_source == "raw":
                    right_mono = right_raw
                else:
                    # 保存拼接前的右单目：raw -> rectified 后的完整极线矫正图。
                    # 注意：这里不加 runtime_right_x_shift / runtime_right_y_shift。
                    # 这两个 shift 是拼接时对右图取样位置的微调，不属于原始 rectified 单目本身。
                    t0 = time.perf_counter()
                    right_mono = rectify_full_frame(right_raw, maps.right_map1, maps.right_map2, maps)
                    build_ms += (time.perf_counter() - t0) * 1000.0
                return right_mono

            if args.preview == "left":
                preview_src = ensure_left_mono()
            elif args.preview == "right":
                preview_src = ensure_right_mono()
            elif args.preview == "wide":
                wide, view = ensure_wide_and_view()
                preview_src = wide
            else:
                if rec_wide.active:
                    wide, view = ensure_wide_and_view()
                else:
                    view = ensure_view_only()
                preview_src = view

            preview = resize_for_display(preview_src, args.display_scale)
            active_names = [name for name, rec in recorders.items() if rec.active]
            active_text = ",".join(active_names) if active_names else "none"
            text = f"REC:{active_text} | 1:left 2:right 3:wide 4:view a:all s:jpg q:quit"
            cv2.putText(preview, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if active_names else (0, 255, 0), 2)

            if not args.headless:
                cv2.imshow("record preview", preview)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            term_key = read_terminal_key()
            if term_key != 255:
                key = term_key

            if key in (ord("q"), 27):
                print("退出。")
                break
            elif key == ord("h"):
                print_key_help()
            elif key == ord("1"):
                start_or_stop_recorder(rec_left, ensure_left_mono())
            elif key == ord("2"):
                start_or_stop_recorder(rec_right, ensure_right_mono())
            elif key == ord("3"):
                wide, view = ensure_wide_and_view()
                start_or_stop_recorder(rec_wide, wide)
            elif key == ord("4"):
                if wide is not None:
                    _, view = ensure_wide_and_view()
                else:
                    view = ensure_view_only()
                start_or_stop_recorder(rec_view, view)
            elif key == ord("a"):
                all_active = all(rec.active for rec in recorders.values())
                if all_active:
                    for rec in recorders.values():
                        rec.stop()
                else:
                    wide, view = ensure_wide_and_view()
                    if not rec_left.active:
                        rec_left.start(ensure_left_mono())
                    if not rec_right.active:
                        rec_right.start(ensure_right_mono())
                    if not rec_wide.active:
                        rec_wide.start(wide)
                    if not rec_view.active:
                        rec_view.start(view)
            elif key == ord("s"):
                ts = current_timestamp()
                ensure_dir(debug_dir)
                cv2.imwrite(os.path.join(debug_dir, f"debug_left_{args.mono_source}_{debug_idx:04d}_{ts}.jpg"), ensure_left_mono(), [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
                cv2.imwrite(os.path.join(debug_dir, f"debug_right_{args.mono_source}_{debug_idx:04d}_{ts}.jpg"), ensure_right_mono(), [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
                wide, view = ensure_wide_and_view()
                cv2.imwrite(os.path.join(debug_dir, f"debug_wide_{debug_idx:04d}_{ts}.jpg"), wide, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
                cv2.imwrite(os.path.join(debug_dir, f"debug_view1920_{debug_idx:04d}_{ts}.jpg"), view, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
                print(f"已保存调试 JPG 到: {debug_dir}")
                debug_idx += 1

            now = time.time()
            if rec_left.active:
                rec_left.enqueue(ensure_left_mono(), now)
            if rec_right.active:
                rec_right.enqueue(ensure_right_mono(), now)
            if rec_wide.active:
                wide, view = ensure_wide_and_view()
                rec_wide.enqueue(wide, now)
            if rec_view.active:
                if wide is not None:
                    _, view = ensure_wide_and_view()
                else:
                    view = ensure_view_only()
                rec_view.enqueue(view, now)

            loop_t1 = time.perf_counter()
            stat_count += 1
            stat_total_ms += (loop_t1 - loop_t0) * 1000.0
            stat_build_ms += build_ms
            if stat_count >= 60:
                avg_total = stat_total_ms / stat_count
                avg_build = stat_build_ms / stat_count
                fps = 1000.0 / max(avg_total, 1e-6)
                print(f"[PROFILE] fps={fps:.1f} | build={avg_build:.1f}ms | total={avg_total:.1f}ms | "
                      f"{rec_left.queue_state()} | {rec_right.queue_state()} | "
                      f"{rec_wide.queue_state()} | {rec_view.queue_state()}")
                stat_count = 0
                stat_total_ms = 0.0
                stat_build_ms = 0.0

    finally:
        print("\n正在清理资源...")
        for rec in recorders.values():
            if rec.active:
                rec.stop()
        if left_cam is not None:
            left_cam.stop()
        if right_cam is not None:
            right_cam.stop()
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        print("清理完成。")


# ============================================================
# 9. 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 h264_rkmpp 多路视频录制脚本")

    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="摄像头采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="禁用摄像头 MJPG，默认使用 MJPG")

    parser.add_argument("--map-file", default=DEFAULT_MAP_FILE, help="双目 remap npz 文件")
    parser.add_argument("--stitch-param", default=DEFAULT_STITCH_PARAM, help="离线拼接参数 npz 文件")

    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT, help="保存根目录")
    parser.add_argument("--left-dir", default=None, help="左单目视频保存目录")
    parser.add_argument("--right-dir", default=None, help="右单目视频保存目录")
    parser.add_argument("--wide-dir", default=None, help="宽幅视频保存目录")
    parser.add_argument("--view-dir", default=None, help="view1920 视频保存目录")
    parser.add_argument("--debug-dir", default=None, help="调试 JPG 保存目录")

    parser.add_argument(
        "--mono-source",
        choices=["rectified", "raw"],
        default="rectified",
        help="左/右单目保存来源：rectified=保存极线矫正后的拼接前单目图；raw=保存摄像头原图",
    )

    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件路径")
    parser.add_argument("--encoder", default="h264_rkmpp", help="FFmpeg 编码器，RK3588 推荐 h264_rkmpp 或 hevc_rkmpp")
    parser.add_argument("--container", default="mp4", help="输出容器后缀，默认 mp4")
    parser.add_argument("--ffmpeg-loglevel", default="error", help="FFmpeg 日志级别：error/warning/info")
    parser.add_argument("--queue-size", type=int, default=8, help="每路写入队列长度")
    parser.add_argument("--gop", type=int, default=60, help="编码 GOP")

    parser.add_argument("--left-fps", type=float, default=20.0, help="左单目保存 FPS")
    parser.add_argument("--right-fps", type=float, default=20.0, help="右单目保存 FPS")
    parser.add_argument("--wide-fps", type=float, default=5.0, help="宽幅保存 FPS，建议 5~10")
    parser.add_argument("--view-fps", type=float, default=20.0, help="view1920 保存 FPS")
    parser.add_argument("--left-bitrate", default="8M", help="左单目码率")
    parser.add_argument("--right-bitrate", default="8M", help="右单目码率")
    parser.add_argument("--wide-bitrate", default="16M", help="宽幅码率")
    parser.add_argument("--view-bitrate", default="8M", help="view1920 码率")

    parser.add_argument("--runtime-seam-x", type=int, default=150, help="接缝中心在 overlap 内的位置")
    parser.add_argument("--runtime-blend-width", type=int, default=40, help="实时融合宽度")
    parser.add_argument("--runtime-right-x-shift", type=int, default=30, help="右图 rectified 取样 x 微调")
    parser.add_argument("--runtime-right-y-shift", type=int, default=-5, help="右图 rectified 取样 y 微调")

    parser.add_argument("--view-width", type=int, default=1920, help="view 输出宽度")
    parser.add_argument("--view-height", type=int, default=1080, help="view 输出高度")
    parser.add_argument("--crop-x", type=int, default=-1, help="view 裁切起点 x，-1 表示居中")
    parser.add_argument("--crop-y", type=int, default=-1, help="view 裁切起点 y，-1 表示居中")

    parser.add_argument("--preview", choices=["left", "right", "wide", "view"], default="view", help="预览哪一路")
    parser.add_argument("--display-scale", type=float, default=0.25, help="预览窗口缩放比例")
    parser.add_argument("--headless", action="store_true", help="不显示窗口，只使用终端按键")
    parser.add_argument("--jpg-quality", type=int, default=95, help="按 s 保存调试 JPG 的质量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("\n================ RK3588 FFmpeg 硬件编码录制 ================")
    print(f"left_device       : {args.left_device}")
    print(f"right_device      : {args.right_device}")
    print(f"camera size       : {args.width} x {args.height}")
    print(f"camera fps        : {args.fps}")
    print(f"map_file          : {args.map_file}")
    print(f"stitch_param      : {args.stitch_param}")
    print(f"use_mjpg          : {not args.no_mjpg}")
    run(args)


if __name__ == "__main__":
    main()
