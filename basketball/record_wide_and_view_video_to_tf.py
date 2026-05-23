#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
record_wide_and_view_video_to_tf.py

用途：
    在 RK3588 / ELF2 / Linux 上，从左右两个 USB 摄像头采集 1920x1080 图像，
    使用已经离线生成好的 stereo_rectify_maps_wide_good.npz 和 stitch_params_good.npz，
    同时录制两个 MP4 文件：

        1. 完整宽幅拼接视频 wide_xxx.mp4
        2. 1920x1080 裁切视频 view1920_xxx.mp4

    两个视频分别保存到两个文件夹中，按键控制开始 / 停止录制。

典型用法：
    python3 record_wide_and_view_video_to_tf.py \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
        --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
        --wide-dir /media/elf/7AC8-E830/basketball_videos/wide \
        --view-dir /media/elf/7AC8-E830/basketball_videos/view1920 \
        --runtime-seam-x 150 \
        --runtime-blend-width 40 \
        --runtime-right-x-shift 30 \
        --runtime-right-y-shift -5 \
        --record-fps 20

        
        cp -r /media/elf/7AC8-E830/basketball_videos/wide /home/elf/work/basketball
按键：
    r / Space : 开始录制 / 停止录制
    s         : 单独保存当前帧 JPG 调试图
    q / Esc   : 退出程序

终端按键：
    如果 OpenCV 窗口没有焦点，也可以在终端输入：
        r 回车    开始 / 停止录制
        s 回车    保存一组 JPG 调试图
        q 回车    退出

注意：
    1. 完整宽幅图的尺寸通常类似 3406x1201。MP4 编码器通常要求宽高为偶数，
       所以本脚本会自动把奇数高度 padding 到偶数，例如 1201 -> 1202。
    2. 宽幅 MP4 分辨率较大，写入 TF 卡压力会比 JPG 单张保存大。
       如果录制卡顿，可以先把 --record-fps 设为 15 或 20。
    3. 退出前请先按 q 正常退出；不要录制时直接拔 TF 卡。
"""

import argparse
import os
import sys
import time
import select
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

# RK3588 / Mali 平台上 OpenCV 有时会尝试启用 OpenCL，
# 可能出现 CL_INVALID_BINARY 或带来额外开销。
# 必须在 import cv2 之前设置。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# ============================================================
# 1. 默认路径和参数
# ============================================================

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"

DEFAULT_MAP_FILE = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz"
DEFAULT_STITCH_PARAM = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz"

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30

DEFAULT_SAVE_ROOT = "/home/elf/work/basketball/video_recordings"


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
# 3. 基础工具函数
# ============================================================

def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def current_timestamp() -> str:
    """生成适合文件名使用的时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    """从 npz 字段中读取图像尺寸。"""
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 格式不正确: {value}")
    return int(flat[0]), int(flat[1])


def get_npz_int(data: np.lib.npyio.NpzFile, key: str, default: Optional[int] = None) -> int:
    """从 npz 中读取 int。"""
    if key not in data.files:
        if default is None:
            raise RuntimeError(f"npz 缺少字段: {key}")
        return int(default)
    return int(np.array(data[key]).reshape(-1)[0])


def convert_maps_for_fast_remap(map1: np.ndarray, map2: np.ndarray):
    """
    把 float32 remap map 转成 fixed-point map。

    这样 cv2.remap 通常会更快。
    """
    if map1.dtype == np.int16 and map2.dtype == np.uint16:
        return map1, map2

    return cv2.convertMaps(
        map1.astype(np.float32),
        map2.astype(np.float32),
        cv2.CV_16SC2,
    )


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    """缩小图像用于 OpenCV 窗口预览。"""
    if abs(scale - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def check_dir_writable(path: str) -> None:
    """检查目录是否可写。"""
    ensure_dir(path)
    test_file = os.path.join(path, ".write_test.tmp")
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise RuntimeError(f"目录不可写: {path}\n原因: {e}")


def read_terminal_key() -> int:
    """
    从终端非阻塞读取按键。

    用途：
        如果 OpenCV 窗口没有焦点，可以在终端输入 r/s/q 后回车。
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


def make_even_for_video(img: np.ndarray) -> np.ndarray:
    """
    保证视频帧宽高为偶数。

    很多 MP4 编码器要求宽高为偶数。
    你的完整宽幅图可能是 3406x1201，高度 1201 是奇数，
    所以这里会在底部 padding 1 行黑边，变成 3406x1202。
    """
    h, w = img.shape[:2]
    pad_right = 1 if (w % 2) else 0
    pad_bottom = 1 if (h % 2) else 0

    if pad_right == 0 and pad_bottom == 0:
        return img

    return cv2.copyMakeBorder(
        img,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def crop_view_from_wide(
    wide: np.ndarray,
    crop_x: int,
    crop_y: int,
    view_w: int,
    view_h: int,
) -> np.ndarray:
    """
    从完整宽幅图中裁出 1920x1080 画面。

    crop_x / crop_y 为 -1 时表示居中裁切。
    这里后续可以接你的自动运镜算法，直接把运镜输出的 crop_x/crop_y 传进来。
    """
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
# 4. 加载 map 和 stitch 参数
# ============================================================

def load_rectify_maps(map_file: str) -> RectifyMaps:
    """加载 stereo_rectify_maps_wide_good.npz。"""
    if not os.path.exists(map_file):
        raise RuntimeError(f"找不到 map 文件: {map_file}")

    data = np.load(map_file)

    required_keys = [
        "left_rect_map1",
        "left_rect_map2",
        "right_rect_map1",
        "right_rect_map2",
        "raw_image_size",
    ]

    for key in required_keys:
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

    left_map1, left_map2 = convert_maps_for_fast_remap(
        data["left_rect_map1"],
        data["left_rect_map2"],
    )
    right_map1, right_map2 = convert_maps_for_fast_remap(
        data["right_rect_map1"],
        data["right_rect_map2"],
    )

    maps = RectifyMaps(
        raw_image_size=raw_image_size,
        rectified_size=rectified_size,
        left_map1=left_map1,
        left_map2=left_map2,
        right_map1=right_map1,
        right_map2=right_map2,
    )

    print("已加载 remap 文件:")
    print(f"  map_file       : {map_file}")
    print(f"  raw_image_size : {maps.raw_image_size}")
    print(f"  rectified_size : {maps.rectified_size}")

    return maps


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
# 5. 摄像头采集
# ============================================================

def open_usb_camera(
    device: str,
    width: int,
    height: int,
    fps: int,
    use_mjpg: bool = True,
) -> cv2.VideoCapture:
    """使用 V4L2 打开 USB 摄像头。"""
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
    """
    后台采集线程。

    用途：
        摄像头 cap.read() 不阻塞主处理循环。
        主线程每次只取最新帧，用于拼接和录制。
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
    """读取左右最新帧。"""
    ok_l, left = left_cam.read_latest()
    ok_r, right = right_cam.read_latest()
    if not ok_l or not ok_r or left is None or right is None:
        return False, None, None
    return True, left, right


# ============================================================
# 6. remap 与宽幅拼接
# ============================================================

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

    这里返回固定大小输出。
    如果 right_x_shift / right_y_shift 导致 ROI 越界，会用黑边补齐，避免尺寸不一致。
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
    maps: RectifyMaps,
    params: StitchParams,
    seam_x: int,
    blend_width: int,
    right_x_shift: int,
    right_y_shift: int,
) -> np.ndarray:
    """从左右 raw 图直接生成完整宽幅拼接图。"""
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

    # 1. 左侧非重叠区域
    left_keep = remap_rectified_roi_fixed_size(
        left_raw,
        maps.left_map1,
        maps.left_map2,
        params.left_keep_x1,
        left_y1,
        params.left_keep_x2,
        left_y2,
    )
    wide[:, left_keep_start:left_keep_end] = left_keep

    # 2. overlap 区域
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

    # overlap 左侧直接取左图，减少大面积重影。
    if blend_x1 > 0:
        left_overlap_left = remap_rectified_roi_fixed_size(
            left_raw,
            maps.left_map1,
            maps.left_map2,
            params.left_overlap_x1,
            left_y1,
            params.left_overlap_x1 + blend_x1,
            left_y2,
        )
        wide[:, overlap_start:overlap_start + blend_x1] = left_overlap_left

    # overlap 右侧直接取右图。
    if blend_x2 < overlap_w:
        right_overlap_right = remap_rectified_roi_fixed_size(
            right_raw,
            maps.right_map1,
            maps.right_map2,
            params.right_overlap_x1 + blend_x2 + right_x_shift,
            right_y1,
            params.right_overlap_x1 + overlap_w + right_x_shift,
            right_y2,
        )
        wide[:, overlap_start + blend_x2:overlap_end] = right_overlap_right

    # seam 附近 alpha 融合。
    left_blend = remap_rectified_roi_fixed_size(
        left_raw,
        maps.left_map1,
        maps.left_map2,
        params.left_overlap_x1 + blend_x1,
        left_y1,
        params.left_overlap_x1 + blend_x2,
        left_y2,
    )

    right_blend = remap_rectified_roi_fixed_size(
        right_raw,
        maps.right_map1,
        maps.right_map2,
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
        maps.right_map1,
        maps.right_map2,
        params.right_keep_x1 + right_x_shift,
        right_y1,
        params.right_keep_x2 + right_x_shift,
        right_y2,
    )
    wide[:, right_keep_start:right_keep_end] = right_keep

    return wide


# ============================================================
# 7. 视频录制器
# ============================================================

class DualVideoRecorder:
    """同时管理宽幅 MP4 和 1920x1080 MP4。"""

    def __init__(self, wide_dir: str, view_dir: str, fps: float, fourcc: str):
        self.wide_dir = wide_dir
        self.view_dir = view_dir
        self.fps = float(fps)
        self.fourcc = fourcc

        self.wide_writer = None
        self.view_writer = None
        self.wide_path = None
        self.view_path = None
        self.frame_count = 0
        self.start_time = None
        self.is_recording = False

    def start(self, wide_frame: np.ndarray, view_frame: np.ndarray):
        """开始一次新录制。"""
        if self.is_recording:
            return

        ensure_dir(self.wide_dir)
        ensure_dir(self.view_dir)

        ts = current_timestamp()
        self.wide_path = os.path.join(self.wide_dir, f"wide_{ts}.mp4")
        self.view_path = os.path.join(self.view_dir, f"view1920_{ts}.mp4")

        wide_video = make_even_for_video(wide_frame)
        view_video = make_even_for_video(view_frame)

        wide_h, wide_w = wide_video.shape[:2]
        view_h, view_w = view_video.shape[:2]

        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)

        self.wide_writer = cv2.VideoWriter(
            self.wide_path,
            fourcc_code,
            self.fps,
            (wide_w, wide_h),
        )
        self.view_writer = cv2.VideoWriter(
            self.view_path,
            fourcc_code,
            self.fps,
            (view_w, view_h),
        )

        if not self.wide_writer.isOpened():
            raise RuntimeError(f"无法打开宽幅视频写入器: {self.wide_path}")
        if not self.view_writer.isOpened():
            raise RuntimeError(f"无法打开 1920 视频写入器: {self.view_path}")

        self.frame_count = 0
        self.start_time = time.time()
        self.is_recording = True

        print("\n开始录制 MP4:")
        print(f"  wide : {self.wide_path}")
        print(f"  view : {self.view_path}")
        print(f"  fps  : {self.fps}")
        print(f"  wide size : {wide_w} x {wide_h}")
        print(f"  view size : {view_w} x {view_h}")

    def write(self, wide_frame: np.ndarray, view_frame: np.ndarray):
        """写入一帧。"""
        if not self.is_recording:
            return

        self.wide_writer.write(make_even_for_video(wide_frame))
        self.view_writer.write(make_even_for_video(view_frame))
        self.frame_count += 1

    def stop(self):
        """停止录制并释放文件。"""
        if not self.is_recording:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0.0

        if self.wide_writer is not None:
            self.wide_writer.release()
        if self.view_writer is not None:
            self.view_writer.release()

        print("\n停止录制 MP4:")
        print(f"  wide : {self.wide_path}")
        print(f"  view : {self.view_path}")
        print(f"  frames: {self.frame_count}")
        print(f"  elapsed: {elapsed:.1f} s")
        if elapsed > 0:
            print(f"  actual write fps: {self.frame_count / elapsed:.1f}")

        self.wide_writer = None
        self.view_writer = None
        self.is_recording = False


# ============================================================
# 8. 主录制流程
# ============================================================

def run(args: argparse.Namespace) -> None:
    maps = load_rectify_maps(args.map_file)
    params = load_stitch_params(args.stitch_param)

    wide_dir = args.wide_dir or os.path.join(args.save_root, "wide")
    view_dir = args.view_dir or os.path.join(args.save_root, "view1920")

    check_dir_writable(wide_dir)
    check_dir_writable(view_dir)

    print("\n运行时参数:")
    print(f"  runtime_seam_x       : {args.runtime_seam_x}")
    print(f"  runtime_blend_width  : {args.runtime_blend_width}")
    print(f"  runtime_right_x_shift: {args.runtime_right_x_shift}")
    print(f"  runtime_right_y_shift: {args.runtime_right_y_shift}")
    print(f"  view size            : {args.view_width} x {args.view_height}")
    print(f"  crop                 : ({args.crop_x}, {args.crop_y})")
    print(f"  record_fps           : {args.record_fps}")
    print(f"  fourcc               : {args.fourcc}")
    print(f"  wide_dir             : {wide_dir}")
    print(f"  view_dir             : {view_dir}")

    cap_left = open_usb_camera(args.left_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)
    cap_right = open_usb_camera(args.right_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)

    left_cam = None
    right_cam = None
    recorder = DualVideoRecorder(wide_dir, view_dir, fps=args.record_fps, fourcc=args.fourcc)

    frame_idx = 0
    stat_count = 0
    stat_total_ms = 0.0
    stat_write_ms = 0.0
    debug_idx = 0

    try:
        # 后台采集可以避免 cap.read 阻塞主循环。
        left_cam = LatestFrameCamera(cap_left, "left").start()
        right_cam = LatestFrameCamera(cap_right, "right").start()

        print("\n等待摄像头第一帧...")
        for _ in range(200):
            ok, _, _ = grab_pair_latest(left_cam, right_cam)
            if ok:
                break
            time.sleep(0.01)

        print("\n开始预览。按键：")
        print("  r / Space : 开始录制 / 停止录制")
        print("  s         : 保存当前 wide/view JPG 调试图")
        print("  q / Esc   : 退出")
        print("")

        while True:
            loop_t0 = time.perf_counter()

            ok, left_raw, right_raw = grab_pair_latest(left_cam, right_cam)
            if not ok:
                print("[警告] 摄像头读取失败")
                time.sleep(0.01)
                continue

            wide = build_wide_stitch_from_raw(
                left_raw=left_raw,
                right_raw=right_raw,
                maps=maps,
                params=params,
                seam_x=args.runtime_seam_x,
                blend_width=args.runtime_blend_width,
                right_x_shift=args.runtime_right_x_shift,
                right_y_shift=args.runtime_right_y_shift,
            )

            view = crop_view_from_wide(
                wide,
                crop_x=args.crop_x,
                crop_y=args.crop_y,
                view_w=args.view_width,
                view_h=args.view_height,
            )

            write_t0 = time.perf_counter()
            if recorder.is_recording:
                recorder.write(wide, view)
            write_t1 = time.perf_counter()

            # 预览默认显示 1920x1080 裁切画面，方便观察最终输出。
            preview_src = view if args.preview == "view" else wide
            preview = resize_for_display(preview_src, args.display_scale)

            status = "REC" if recorder.is_recording else "STANDBY"
            text = (
                f"{status} | r: start/stop | s: jpg | q: quit | "
                f"frames:{recorder.frame_count if recorder.is_recording else 0}"
            )
            cv2.putText(
                preview,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if recorder.is_recording else (0, 255, 0),
                2,
            )

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

            if key in (ord("r"), ord(" ")):
                if recorder.is_recording:
                    recorder.stop()
                else:
                    recorder.start(wide, view)

            if key == ord("s"):
                ts = current_timestamp()
                wide_jpg = os.path.join(wide_dir, f"debug_wide_{debug_idx:04d}_{ts}.jpg")
                view_jpg = os.path.join(view_dir, f"debug_view1920_{debug_idx:04d}_{ts}.jpg")
                cv2.imwrite(wide_jpg, wide, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
                cv2.imwrite(view_jpg, view, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
                print("已保存调试 JPG:")
                print(f"  {wide_jpg}")
                print(f"  {view_jpg}")
                debug_idx += 1

            loop_t1 = time.perf_counter()

            stat_count += 1
            stat_total_ms += (loop_t1 - loop_t0) * 1000.0
            stat_write_ms += (write_t1 - write_t0) * 1000.0
            frame_idx += 1

            if stat_count >= 60:
                avg_total = stat_total_ms / stat_count
                avg_write = stat_write_ms / stat_count
                fps = 1000.0 / max(avg_total, 1e-6)
                print(
                    f"[PROFILE] fps={fps:.1f} | "
                    f"write={avg_write:.1f}ms | "
                    f"total={avg_total:.1f}ms | "
                    f"recording={recorder.is_recording}"
                )
                stat_count = 0
                stat_total_ms = 0.0
                stat_write_ms = 0.0

    finally:
        if recorder.is_recording:
            recorder.stop()
        if left_cam is not None:
            left_cam.stop()
        if right_cam is not None:
            right_cam.stop()
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


# ============================================================
# 9. 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record full wide stitched MP4 and 1920x1080 view MP4."
    )

    # 摄像头参数
    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="摄像头采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="禁用 MJPG，默认使用 MJPG")

    # 离线文件
    parser.add_argument("--map-file", default=DEFAULT_MAP_FILE, help="双目 remap npz 文件")
    parser.add_argument("--stitch-param", default=DEFAULT_STITCH_PARAM, help="离线拼接参数 npz 文件")

    # 保存目录
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT, help="保存根目录，会自动创建 wide/ 和 view1920/")
    parser.add_argument("--wide-dir", default=None, help="宽幅 MP4 保存目录，优先级高于 --save-root")
    parser.add_argument("--view-dir", default=None, help="1920x1080 MP4 保存目录，优先级高于 --save-root")

    # 视频参数
    parser.add_argument("--record-fps", type=float, default=20.0, help="写入 MP4 的帧率。宽幅视频压力大，建议 15~20 起步")
    parser.add_argument("--fourcc", default="mp4v", help="视频编码 fourcc，默认 mp4v")
    parser.add_argument("--jpg-quality", type=int, default=95, help="按 s 保存调试 JPG 的质量")

    # 1920x1080 裁切参数
    parser.add_argument("--view-width", type=int, default=1920, help="裁切输出宽度")
    parser.add_argument("--view-height", type=int, default=1080, help="裁切输出高度")
    parser.add_argument("--crop-x", type=int, default=-1, help="宽幅图裁切起点 x，-1 表示居中")
    parser.add_argument("--crop-y", type=int, default=-1, help="宽幅图裁切起点 y，-1 表示居中")

    # 拼接微调参数：默认使用你当前验证过可接受的参数
    parser.add_argument("--runtime-seam-x", type=int, default=150, help="接缝中心在 overlap 内的位置")
    parser.add_argument("--runtime-blend-width", type=int, default=40, help="实时融合宽度，重影明显时不要太大")
    parser.add_argument("--runtime-right-x-shift", type=int, default=30, help="右图 rectified 取样 x 微调")
    parser.add_argument("--runtime-right-y-shift", type=int, default=-5, help="右图 rectified 取样 y 微调")

    # 预览参数
    parser.add_argument("--display-scale", type=float, default=0.25, help="预览窗口缩放比例")
    parser.add_argument("--preview", choices=["view", "wide"], default="view", help="预览 view 或 wide")
    parser.add_argument("--headless", action="store_true", help="不显示 OpenCV 窗口，只用终端 r/s/q 控制")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ 录制双目拼接 MP4 ================")
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
