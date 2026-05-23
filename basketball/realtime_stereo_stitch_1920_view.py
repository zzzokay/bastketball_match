#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
realtime_stereo_stitch.py

用途：
    实时双目矫正 + 拼接脚本。

对应你的完整流程中的实时阶段：

    1. 读取左右 RGB 图
    2. 用离线 map 做 remap
    3. 根据离线 overlap_px 裁剪
    4. 根据离线 vertical_offset 平移
    5. 做亮度 / 颜色补偿
    6. 用离线 alpha mask 融合
    7. 输出拼接图

依赖的离线文件：
    1. stereo_rectify_maps_wide.npz
        由 offline_build_stereo_rectify_maps.py 生成。
        保存 raw -> rectified 的 remap 查找表。

    2. stitch_params.npz
        由 offline_estimate_stitch_params.py 生成。
        保存 overlap_px / vertical_offset / alpha_mask / 裁剪 ROI 等参数。

典型用法：
    测试帧率：
    python3 realtime_stereo_stitch_1920_view.py \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
        --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
        --width 1920 \
        --height 1080 \
        --fps 30 \
        --view-width 1920 \
        --view-height 1080 \
        --crop-x -1 \
        --crop-y -1 \
        --display-scale 0.25 \
        --disable-color-balance \
        --threaded-capture

调试时，可以通过调整    --runtime-right-x-shift 和--runtime-right-y-shift 
        解决重影问题。
        
python3 realtime_stereo_stitch_1920_view.py \
    --left-device /dev/video41 \
    --right-device /dev/video43 \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
    --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --view-width 1920 \
    --view-height 1080 \
    --crop-x -1 \
    --crop-y -1 \
    --display-scale 0.25 \
    --disable-color-balance \
    --threaded-capture \
    --runtime-seam-x 150 \
    --runtime-blend-width 40 \
    --runtime-right-x-shift 30 \
    --runtime-right-y-shift -5

    
按键：
    q / Esc ：退出
    s       ：保存当前帧调试图
    b       ：开启/关闭亮度颜色补偿
    l       ：开启/关闭水平参考线
    r       ：开启/关闭显示左右 rectified 调试图
"""

import argparse
import os
import time
import sys
import select
from dataclasses import dataclass
from typing import Tuple, Optional

import threading

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

DEFAULT_MAP_FILE = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz"
DEFAULT_STITCH_PARAM = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params.npz"

DEFAULT_SAVE_DIR = "/home/elf/work/basketball/realtime_stitch_debug"

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class StitchParams:
    """
    离线拼接参数。

    这些参数来自 stitch_params.npz。

    overlap_px:
        左右图重叠区域宽度。

    vertical_offset:
        右图相对于左图的垂直偏移。
        > 0 表示右图向下。
        < 0 表示右图向上。

    alpha_mask:
        融合 mask。
        形状通常是 [common_h, overlap_px]。
        0 表示使用左图。
        1 表示使用右图。
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

    alpha_mask: np.ndarray


@dataclass
class RectifyMaps:
    """
    双目极线矫正 remap 查找表。

    实时阶段使用 raw -> rectified map：

        left_rect_map1 / left_rect_map2
        right_rect_map1 / right_rect_map2

    也就是说：
        摄像头原始图像
            ↓ cv2.remap
        双目极线矫正后的图像
    """

    raw_image_size: Tuple[int, int]
    rectified_size: Tuple[int, int]

    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


class FPSCounter:
    """实时 FPS 统计器。"""

    def __init__(self):
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def update(self) -> float:
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_time

        if dt >= 1.0:
            self.fps = self.frame_count / dt
            self.frame_count = 0
            self.last_time = now

        return self.fps


# ============================================================
# 3. 基础工具函数
# ============================================================

def ensure_dir(path: str) -> None:
    """创建目录。"""
    os.makedirs(path, exist_ok=True)


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    """
    从 npz 里读取图像尺寸。

    支持：
        [width, height]
        [[width, height]]
        numpy int 类型
    """
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 格式不正确: {value}")
    return int(flat[0]), int(flat[1])

def convert_maps_for_fast_remap(map1: np.ndarray, map2: np.ndarray):
    """
    把 float32 remap map 转成 OpenCV 更快的 fixed-point map。
    """
    if map1.dtype == np.int16 and map2.dtype == np.uint16:
        return map1, map2

    return cv2.convertMaps(
        map1.astype(np.float32),
        map2.astype(np.float32),
        cv2.CV_16SC2,
    )

def get_npz_int(data, key: str, default: Optional[int] = None) -> int:
    """
    从 npz 中读取 int。

    如果字段不存在并且提供了 default，就返回 default。
    """
    if key not in data.files:
        if default is None:
            raise RuntimeError(f"stitch_params.npz 缺少字段: {key}")
        return int(default)

    return int(np.array(data[key]).reshape(-1)[0])


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    """缩小显示图，降低显示开销。"""
    if abs(scale - 1.0) < 1e-6:
        return img

    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def read_terminal_key() -> int:
    """
    从终端非阻塞读取按键。

    为什么需要这个：
        cv2.waitKey() 只能读取 OpenCV 图像窗口里的按键。
        如果你在 SSH / VSCode 终端里输入 s，原脚本读不到。

    使用方法：
        在终端输入：
            s 回车
        就可以触发保存。
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


def draw_horizontal_lines(img: np.ndarray, step: int = 80) -> np.ndarray:
    """
    在图像上画水平参考线，用来观察左右极线是否大致水平。
    """
    out = img.copy()
    h, w = out.shape[:2]

    for y in range(0, h, max(1, step)):
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)

    return out


# ============================================================
# 4. 加载离线文件
# ============================================================

def load_rectify_maps(map_file: str) -> RectifyMaps:
    """
    加载 stereo_rectify_maps_wide.npz。

    注意：
        实时阶段必须使用 raw -> rectified 的 map。
        也就是：
            left_rect_map1 / left_rect_map2
            right_rect_map1 / right_rect_map2

        不要使用 left_undist_rect_map1 这类调试 map，
        否则会重复矫正。
    """
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


def create_alpha_mask(height: int, overlap_px: int, blend_width: int) -> np.ndarray:
    """
    重新生成 alpha mask。

    只有在 stitch_params.npz 中没有 alpha_mask，
    或 alpha_mask 尺寸不匹配时才会用到。
    """
    overlap_px = int(overlap_px)
    blend_width = int(blend_width)

    if overlap_px <= 1:
        return np.ones((height, 1), dtype=np.float32)

    blend_width = max(1, min(blend_width, overlap_px))

    alpha = np.zeros((height, overlap_px), dtype=np.float32)

    blend_x1 = (overlap_px - blend_width) // 2
    blend_x2 = blend_x1 + blend_width

    alpha[:, :blend_x1] = 0.0

    ramp = np.linspace(0.0, 1.0, blend_width, dtype=np.float32)
    alpha[:, blend_x1:blend_x2] = ramp.reshape(1, -1)

    alpha[:, blend_x2:] = 1.0

    return alpha


def load_stitch_params(stitch_param_file: str) -> StitchParams:
    """
    加载 stitch_params.npz。

    这个文件由 offline_estimate_stitch_params.py 生成。
    """
    if not os.path.exists(stitch_param_file):
        raise RuntimeError(f"找不到 stitch 参数文件: {stitch_param_file}")

    data = np.load(stitch_param_file)

    overlap_px = get_npz_int(data, "overlap_px")
    vertical_offset = get_npz_int(data, "vertical_offset")
    blend_width = get_npz_int(data, "blend_width", default=overlap_px)

    left_y1 = get_npz_int(data, "left_y1")
    left_y2 = get_npz_int(data, "left_y2")
    right_y1 = get_npz_int(data, "right_y1")
    right_y2 = get_npz_int(data, "right_y2")

    left_keep_x1 = get_npz_int(data, "left_keep_x1")
    left_keep_x2 = get_npz_int(data, "left_keep_x2")
    left_overlap_x1 = get_npz_int(data, "left_overlap_x1")
    left_overlap_x2 = get_npz_int(data, "left_overlap_x2")

    right_overlap_x1 = get_npz_int(data, "right_overlap_x1")
    right_overlap_x2 = get_npz_int(data, "right_overlap_x2")
    right_keep_x1 = get_npz_int(data, "right_keep_x1")
    right_keep_x2 = get_npz_int(data, "right_keep_x2")

    output_width = get_npz_int(data, "output_width")
    output_height = get_npz_int(data, "output_height")

    common_h = left_y2 - left_y1

    if "alpha_mask" in data.files:
        alpha_mask = data["alpha_mask"].astype(np.float32)
    else:
        alpha_mask = create_alpha_mask(common_h, overlap_px, blend_width)

    # 如果 alpha 尺寸不匹配，重新生成。
    if alpha_mask.shape[0] != common_h or alpha_mask.shape[1] != overlap_px:
        print("[警告] alpha_mask 尺寸与当前 ROI 不匹配，自动重新生成。")
        alpha_mask = create_alpha_mask(common_h, overlap_px, blend_width)

    params = StitchParams(
        overlap_px=overlap_px,
        vertical_offset=vertical_offset,
        blend_width=blend_width,

        left_y1=left_y1,
        left_y2=left_y2,
        right_y1=right_y1,
        right_y2=right_y2,

        left_keep_x1=left_keep_x1,
        left_keep_x2=left_keep_x2,
        left_overlap_x1=left_overlap_x1,
        left_overlap_x2=left_overlap_x2,

        right_overlap_x1=right_overlap_x1,
        right_overlap_x2=right_overlap_x2,
        right_keep_x1=right_keep_x1,
        right_keep_x2=right_keep_x2,

        output_width=output_width,
        output_height=output_height,

        alpha_mask=alpha_mask,
    )

    print("\n已加载拼接参数:")
    print(f"  stitch_param    : {stitch_param_file}")
    print(f"  overlap_px      : {params.overlap_px}")
    print(f"  vertical_offset : {params.vertical_offset}")
    print(f"  blend_width     : {params.blend_width}")
    print(f"  output_size     : {params.output_width} x {params.output_height}")
    print(f"  left_y          : {params.left_y1} -> {params.left_y2}")
    print(f"  right_y         : {params.right_y1} -> {params.right_y2}")
    print(f"  alpha shape     : {params.alpha_mask.shape}")

    return params


# ============================================================
# 5. 摄像头打开和读取
# ============================================================

def open_usb_camera(
    device: str,
    width: int,
    height: int,
    fps: int,
    use_mjpg: bool = True,
) -> cv2.VideoCapture:
    """
    使用 V4L2 打开 USB 摄像头。

    为什么默认使用 MJPG？
        1920x1080@30fps 双路摄像头如果使用 YUYV，
        USB 带宽压力非常大。
        MJPG 是压缩格式，通常更容易跑到高帧率。

    注意：
        设置 FOURCC / 分辨率 / FPS 后，实际值要通过 cap.get 再检查。
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device}")

    # 尽量减少缓冲，降低实时延迟。
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
    fourcc_str = "".join(
        chr((fourcc_int >> 8 * i) & 0xFF)
        for i in range(4)
    )

    print(f"\n摄像头已打开: {device}")
    print(f"  request size : {width}x{height}")
    print(f"  real size    : {real_w}x{real_h}")
    print(f"  request fps  : {fps}")
    print(f"  real fps     : {real_fps}")
    print(f"  fourcc       : {fourcc_str}")

    if (real_w, real_h) != (width, height):
        print("[警告] 实际分辨率和请求分辨率不同。")
        print("       实时输入尺寸必须和生成 map 的 raw_image_size 一致，否则会 resize，影响速度和精度。")

    return cap


class LatestFrameCamera:
    """
    后台采集线程。

    目的：
        摄像头 grab/retrieve 不再阻塞主处理循环。
        主线程每次只拿“最新的一帧”。

    这样可以把：
        摄像头读取 + 图像处理
    从串行变成流水线。
    """

    def __init__(self, cap: cv2.VideoCapture, name: str = "camera"):
        self.cap = cap
        self.name = name

        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )

    def start(self):
        self.thread.start()
        return self

    def _worker(self):
        while not self.stopped:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.ok = False
                time.sleep(0.002)
                continue

            with self.lock:
                self.frame = frame
                self.ok = True

    def read_latest(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None

            # 这里先不 copy，减少开销。
            # OpenCV read 每次会返回新的 ndarray，一般不会被后台线程原地修改。
            return True, self.frame

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass


def grab_pair(
    cap_left: cv2.VideoCapture,
    cap_right: cv2.VideoCapture,
    use_grab_retrieve: bool,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    读取一对左右图。

    use_grab_retrieve=True：
        先同时 grab 左右帧，再 retrieve 解码。
        这样左右帧时间差通常比连续 read 更小。

    use_grab_retrieve=False：
        直接 cap.read。
    """
    if use_grab_retrieve:
        ok_l = cap_left.grab()
        ok_r = cap_right.grab()

        if not ok_l or not ok_r:
            return False, None, None

        ret_l, frame_l = cap_left.retrieve()
        ret_r, frame_r = cap_right.retrieve()
    else:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

    if not ret_l or not ret_r or frame_l is None or frame_r is None:
        return False, None, None

    return True, frame_l, frame_r

def grab_pair_latest(
    left_cam: LatestFrameCamera,
    right_cam: LatestFrameCamera,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从两个后台采集线程中取最新帧。
    """
    ok_l, frame_l = left_cam.read_latest()
    ok_r, frame_r = right_cam.read_latest()

    if not ok_l or not ok_r or frame_l is None or frame_r is None:
        return False, None, None

    return True, frame_l, frame_r

# ============================================================
# 6. remap 与拼接
# ============================================================

def rectify_raw_pair(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    maps: RectifyMaps,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对原始左右图做双目极线矫正。

    实时阶段只做一次 remap：
        raw -> rectified

    这样避免：
        raw -> 单目矫正 -> 双目极线矫正
    这种两次插值带来的速度下降和画质损失。
    """
    raw_w, raw_h = maps.raw_image_size

    # 如果摄像头实际输出尺寸不等于 map 需要的 raw_image_size，
    # 这里自动 resize。
    # 但最好不要依赖这个功能，应该让摄像头直接输出正确尺寸。
    if (left_raw.shape[1], left_raw.shape[0]) != maps.raw_image_size:
        left_raw = cv2.resize(left_raw, maps.raw_image_size)

    if (right_raw.shape[1], right_raw.shape[0]) != maps.raw_image_size:
        right_raw = cv2.resize(right_raw, maps.raw_image_size)

    left_rect = cv2.remap(
        left_raw,
        maps.left_map1,
        maps.left_map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    right_rect = cv2.remap(
        right_raw,
        maps.right_map1,
        maps.right_map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return left_rect, right_rect

def remap_rectified_roi(
    raw: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """
    只 remap rectified 坐标系中的一个 ROI。

    关键点：
        map1/map2 的坐标系是 rectified 输出坐标系。
        直接切 map1[y1:y2, x1:x2]，cv2.remap 输出的就是该 rectified ROI。
    """
    map_h, map_w = map1.shape[:2]

    x1 = int(max(0, min(map_w, x1)))
    x2 = int(max(0, min(map_w, x2)))
    y1 = int(max(0, min(map_h, y1)))
    y2 = int(max(0, min(map_h, y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return cv2.remap(
        raw,
        map1[y1:y2, x1:x2],
        map2[y1:y2, x1:x2],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def stitch_view_roi_remap(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    maps: RectifyMaps,
    params: StitchParams,
    crop_x: int,
    crop_y: int,
    view_w: int = 1920,
    view_h: int = 1080,
    seam_x: int = -1,
    right_x_shift: int = 0,
    right_y_shift: int = 0,
) -> np.ndarray:
    """
    直接从虚拟宽图坐标系裁出 1920x1080 视口。

    和上一版不同：
        上一版：先完整 remap 左右图，再裁 1920x1080。
        这一版：反推 1920x1080 视口需要哪些 rectified ROI，只 remap 这些 ROI。
    """

    # 如果摄像头实际尺寸和 map 的 raw_image_size 不一致，仍然兜底 resize。
    # 但正常情况下不应该走到这里。
    if (left_raw.shape[1], left_raw.shape[0]) != maps.raw_image_size:
        left_raw = cv2.resize(left_raw, maps.raw_image_size)

    if (right_raw.shape[1], right_raw.shape[0]) != maps.raw_image_size:
        right_raw = cv2.resize(right_raw, maps.raw_image_size)

    output_w = int(params.output_width)
    output_h = int(params.output_height)

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

    # 虚拟宽图中的三个区间
    left_keep_start = 0
    left_keep_end = left_keep_w

    overlap_start = left_keep_end
    overlap_end = overlap_start + overlap_w

    right_keep_start = overlap_end
    right_keep_end = right_keep_start + right_keep_w

    view_start = crop_x
    view_end = crop_x + view_w

    def paste_direct(
        seg_start: int,
        seg_end: int,
        raw: np.ndarray,
        map1: np.ndarray,
        map2: np.ndarray,
        rect_x_base: int,
        rect_y1: int,
        rect_y2: int,
    ):
        """
        把虚拟宽图中的一个普通区间 remap 后贴到 view。
        """
        ix1 = max(view_start, seg_start)
        ix2 = min(view_end, seg_end)

        if ix2 <= ix1:
            return

        dst_x1 = ix1 - view_start
        dst_x2 = ix2 - view_start

        rect_x1 = rect_x_base + (ix1 - seg_start)
        rect_x2 = rect_x_base + (ix2 - seg_start)

        roi = remap_rectified_roi(
            raw,
            map1,
            map2,
            rect_x1,
            rect_y1,
            rect_x2,
            rect_y2,
        )

        if roi is not None:
            view[:valid_h, dst_x1:dst_x2] = roi

    # 1. 左侧非重叠区域：只 remap 视口覆盖到的部分
    paste_direct(
        left_keep_start,
        left_keep_end,
        left_raw,
        maps.left_map1,
        maps.left_map2,
        params.left_keep_x1,
        left_y1,
        left_y2,
    )

    # 2. overlap 区域
    ix1 = max(view_start, overlap_start)
    ix2 = min(view_end, overlap_end)

    if ix2 > ix1:
        ox1 = ix1 - overlap_start
        ox2 = ix2 - overlap_start

        blend_width = max(1, min(int(params.blend_width), overlap_w))

        # seam_x 表示接缝中心在 overlap 内部的位置。
        # -1 表示默认放在 overlap 中间。
        if seam_x < 0:
            seam_x_used = overlap_w // 2
        else:
            seam_x_used = int(seam_x)

        # 防止接缝太靠边导致 blend 区域越界
        half_blend = blend_width // 2
        seam_x_used = int(np.clip(
            seam_x_used,
            half_blend,
            overlap_w - (blend_width - half_blend),
        ))

        blend_x1 = seam_x_used - half_blend
        blend_x2 = blend_x1 + blend_width

        def paste_overlap_left(local_x1: int, local_x2: int):
            if local_x2 <= local_x1:
                return

            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start

            rect_x1 = params.left_overlap_x1 + local_x1
            rect_x2 = params.left_overlap_x1 + local_x2

            roi = remap_rectified_roi(
                left_raw,
                maps.left_map1,
                maps.left_map2,
                rect_x1,
                left_y1,
                rect_x2,
                left_y2,
            )

            if roi is not None:
                view[:valid_h, dst_x1:dst_x2] = roi

        def paste_overlap_right(local_x1: int, local_x2: int):
            if local_x2 <= local_x1:
                return

            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start

            rect_x1 = params.right_overlap_x1 + local_x1 + right_x_shift
            rect_x2 = params.right_overlap_x1 + local_x2 + right_x_shift
            roi = remap_rectified_roi(
                right_raw,
                maps.right_map1,
                maps.right_map2,
                rect_x1,
                right_y1,
                rect_x2,
                right_y2,
            )

            if roi is not None:
                view[:valid_h, dst_x1:dst_x2] = roi

        # overlap 左半边直接用左图
        paste_overlap_left(ox1, min(ox2, blend_x1))

        # overlap 右半边直接用右图
        paste_overlap_right(max(ox1, blend_x2), ox2)

        # 中间 blend_width 区域才做左右 remap + alpha 融合
        bx1 = max(ox1, blend_x1)
        bx2 = min(ox2, blend_x2)

        if bx2 > bx1:
            dst_x1 = overlap_start + bx1 - view_start
            dst_x2 = overlap_start + bx2 - view_start

            left_roi = remap_rectified_roi(
                left_raw,
                maps.left_map1,
                maps.left_map2,
                params.left_overlap_x1 + bx1,
                left_y1,
                params.left_overlap_x1 + bx2,
                left_y2,
            )

            right_roi = remap_rectified_roi(
                right_raw,
                maps.right_map1,
                maps.right_map2,
                params.right_overlap_x1 + bx1 + right_x_shift,
                right_y1,
                params.right_overlap_x1 + bx2 + right_x_shift,
                right_y2,
            )

            if left_roi is not None and right_roi is not None:
                alpha_line = np.linspace(0.0, 1.0, blend_width, dtype=np.float32)
                alpha = alpha_line[bx1 - blend_x1:bx2 - blend_x1].reshape(1, -1, 1)

                blended = (
                    left_roi.astype(np.float32) * (1.0 - alpha)
                    + right_roi.astype(np.float32) * alpha
                )

                view[:valid_h, dst_x1:dst_x2] = np.clip(
                    blended,
                    0,
                    255,
                ).astype(np.uint8)

    # 3. 右侧非重叠区域：只 remap 视口覆盖到的部分
    paste_direct(
        right_keep_start,
        right_keep_end,
        right_raw,
        maps.right_map1,
        maps.right_map2,
        params.right_keep_x1 + right_x_shift,
        right_y1,
        right_y2,
    )

    return view


def estimate_rgb_gain_from_overlap(
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
    clip_min: float = 0.75,
    clip_max: float = 1.25,
) -> np.ndarray:
    """
    根据重叠区域估计右图到左图的 RGB 增益。

    目标：
        让右图颜色和亮度尽量接近左图，减轻拼接缝。

    为什么只用 overlap 区域？
        拼接缝处最重要的是左右图在重叠区域颜色接近。
        全图统计可能会被左右不同内容误导。

    返回：
        gain，形如 [B_gain, G_gain, R_gain]
        注意 OpenCV 图像通道顺序是 BGR，不是 RGB。
    """
    left_f = left_overlap.astype(np.float32)
    right_f = right_overlap.astype(np.float32)

    # 过滤过暗和过曝区域，避免黑边、灯管过曝影响统计。
    left_gray = cv2.cvtColor(left_overlap, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_overlap, cv2.COLOR_BGR2GRAY)

    valid = (
        (left_gray > 20) & (left_gray < 245) &
        (right_gray > 20) & (right_gray < 245)
    )

    if valid.mean() < 0.05:
        # 有效像素太少，返回不调整。
        return np.ones(3, dtype=np.float32)

    left_pixels = left_f[valid]
    right_pixels = right_f[valid]

    mean_l = left_pixels.mean(axis=0)
    mean_r = right_pixels.mean(axis=0)

    gain = mean_l / (mean_r + 1e-6)
    gain = np.clip(gain, clip_min, clip_max).astype(np.float32)

    return gain


def apply_gain(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """
    对图像应用 BGR 三通道增益。
    """
    out = img.astype(np.float32) * gain.reshape(1, 1, 3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def stitch_rectified_pair(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    params: StitchParams,
    enable_color_balance: bool,
    last_gain: Optional[np.ndarray],
    gain_smooth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据离线参数拼接左右 rectified 图。

    返回：
        stitched:
            拼接输出图。

        gain:
            当前使用的 BGR 增益。
            用于下一帧平滑，避免颜色补偿一闪一闪。
    """
    # --------------------------------------------------------
    # 1. 按离线 ROI 裁剪左右图
    # --------------------------------------------------------

    left_keep = left_rect[
        params.left_y1:params.left_y2,
        params.left_keep_x1:params.left_keep_x2,
    ]

    left_overlap = left_rect[
        params.left_y1:params.left_y2,
        params.left_overlap_x1:params.left_overlap_x2,
    ]

    right_overlap = right_rect[
        params.right_y1:params.right_y2,
        params.right_overlap_x1:params.right_overlap_x2,
    ]

    right_keep = right_rect[
        params.right_y1:params.right_y2,
        params.right_keep_x1:params.right_keep_x2,
    ]

    # 安全检查，避免参数不匹配导致崩溃。
    if left_overlap.shape[:2] != right_overlap.shape[:2]:
        raise RuntimeError(
            f"左右 overlap 尺寸不一致: left={left_overlap.shape}, right={right_overlap.shape}"
        )

    if left_overlap.shape[1] != params.overlap_px:
        raise RuntimeError(
            f"overlap 宽度与参数不一致: image={left_overlap.shape[1]}, param={params.overlap_px}"
        )

    # --------------------------------------------------------
    # 2. 亮度 / 颜色补偿
    # --------------------------------------------------------
    # 只调整右图，让右图的重叠区域接近左图。
    # 如果你后续发现颜色补偿导致闪烁，可以启动时加 --disable-color-balance。
    if enable_color_balance:
        current_gain = estimate_rgb_gain_from_overlap(left_overlap, right_overlap)

        if last_gain is None:
            gain = current_gain
        else:
            # 指数平滑：
            # gain_smooth 越大，变化越慢，越不容易闪烁。
            gain = gain_smooth * last_gain + (1.0 - gain_smooth) * current_gain
            gain = gain.astype(np.float32)

        right_overlap_used = apply_gain(right_overlap, gain)
        right_keep_used = apply_gain(right_keep, gain)
    else:
        gain = np.ones(3, dtype=np.float32)
        right_overlap_used = right_overlap
        right_keep_used = right_keep

    # --------------------------------------------------------
    # 3. alpha 融合
    # --------------------------------------------------------
    alpha = params.alpha_mask

    if alpha.shape[:2] != left_overlap.shape[:2]:
        # 理论上不应该发生。
        # 如果发生，说明 stitch_params 和当前图像尺寸不匹配。
        alpha = cv2.resize(
            alpha,
            (left_overlap.shape[1], left_overlap.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

    alpha3 = alpha[:, :, None]

    blended = (
        left_overlap.astype(np.float32) * (1.0 - alpha3)
        + right_overlap_used.astype(np.float32) * alpha3
    )

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # --------------------------------------------------------
    # 4. 拼接最终输出
    # --------------------------------------------------------
    stitched = np.hstack([
        left_keep,
        blended,
        right_keep_used,
    ])

    return stitched, gain

def stitch_rectified_view_1920(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    params: StitchParams,
    crop_x: int,
    crop_y: int,
    view_w: int = 1920,
    view_h: int = 1080,
) -> np.ndarray:
    """
    不生成完整 stitched 宽图。
    直接从虚拟 stitched 坐标系中裁出 view_w x view_h 的主画面。

    虚拟宽图结构：
        [left_keep][overlap][right_keep]
    """

    output_w = int(params.output_width)
    output_h = int(params.output_height)

    crop_x = int(np.clip(crop_x, 0, max(0, output_w - view_w)))
    crop_y = int(np.clip(crop_y, 0, max(0, output_h - view_h)))

    view = np.zeros((view_h, view_w, 3), dtype=np.uint8)

    valid_h = min(view_h, output_h - crop_y)
    if valid_h <= 0:
        return view

    left_y1 = params.left_y1 + crop_y
    left_y2 = left_y1 + valid_h
    right_y1 = params.right_y1 + crop_y
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

    def copy_region(seg_start, seg_end, src, src_y1, src_y2, src_x_base):
        ix1 = max(view_start, seg_start)
        ix2 = min(view_end, seg_end)
        if ix2 <= ix1:
            return

        dst_x1 = ix1 - view_start
        dst_x2 = ix2 - view_start

        src_x1 = src_x_base + (ix1 - seg_start)
        src_x2 = src_x_base + (ix2 - seg_start)

        view[:valid_h, dst_x1:dst_x2] = src[src_y1:src_y2, src_x1:src_x2]

    # 1. 左侧非重叠区域，直接来自左图
    copy_region(
        left_keep_start,
        left_keep_end,
        left_rect,
        left_y1,
        left_y2,
        params.left_keep_x1,
    )

    # 2. overlap 区域：只在 blend_width 那一小段做浮点融合
    ix1 = max(view_start, overlap_start)
    ix2 = min(view_end, overlap_end)

    if ix2 > ix1:
        ox1 = ix1 - overlap_start
        ox2 = ix2 - overlap_start

        blend_width = max(1, min(int(params.blend_width), overlap_w))
        blend_x1 = (overlap_w - blend_width) // 2
        blend_x2 = blend_x1 + blend_width

        def copy_overlap_left(local_x1, local_x2):
            if local_x2 <= local_x1:
                return

            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start

            sx1 = params.left_overlap_x1 + local_x1
            sx2 = params.left_overlap_x1 + local_x2

            view[:valid_h, dst_x1:dst_x2] = left_rect[left_y1:left_y2, sx1:sx2]

        def copy_overlap_right(local_x1, local_x2):
            if local_x2 <= local_x1:
                return

            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start

            sx1 = params.right_overlap_x1 + local_x1
            sx2 = params.right_overlap_x1 + local_x2

            view[:valid_h, dst_x1:dst_x2] = right_rect[right_y1:right_y2, sx1:sx2]

        # overlap 左半部分直接用左图
        copy_overlap_left(ox1, min(ox2, blend_x1))

        # overlap 右半部分直接用右图
        copy_overlap_right(max(ox1, blend_x2), ox2)

        # 中间真正融合带
        bx1 = max(ox1, blend_x1)
        bx2 = min(ox2, blend_x2)

        if bx2 > bx1:
            dst_x1 = overlap_start + bx1 - view_start
            dst_x2 = overlap_start + bx2 - view_start

            l_sx1 = params.left_overlap_x1 + bx1
            l_sx2 = params.left_overlap_x1 + bx2

            r_sx1 = params.right_overlap_x1 + bx1
            r_sx2 = params.right_overlap_x1 + bx2

            alpha_line = np.linspace(0.0, 1.0, blend_width, dtype=np.float32)
            alpha = alpha_line[bx1 - blend_x1:bx2 - blend_x1].reshape(1, -1, 1)

            left_part = left_rect[left_y1:left_y2, l_sx1:l_sx2].astype(np.float32)
            right_part = right_rect[right_y1:right_y2, r_sx1:r_sx2].astype(np.float32)

            view[:valid_h, dst_x1:dst_x2] = np.clip(
                left_part * (1.0 - alpha) + right_part * alpha,
                0,
                255,
            ).astype(np.uint8)

    # 3. 右侧非重叠区域，直接来自右图
    copy_region(
        right_keep_start,
        right_keep_end,
        right_rect,
        right_y1,
        right_y2,
        params.right_keep_x1,
    )

    return view


# ============================================================
# 7. 保存调试图
# ============================================================

def save_debug_frames(
    save_dir: str,
    idx: int,
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    stitched: np.ndarray,
) -> None:
    """
    保存当前帧调试图。
    """
    ensure_dir(save_dir)

    cv2.imwrite(os.path.join(save_dir, f"frame_{idx:06d}_left_raw.jpg"), left_raw)
    cv2.imwrite(os.path.join(save_dir, f"frame_{idx:06d}_right_raw.jpg"), right_raw)
    cv2.imwrite(os.path.join(save_dir, f"frame_{idx:06d}_left_rect.jpg"), left_rect)
    cv2.imwrite(os.path.join(save_dir, f"frame_{idx:06d}_right_rect.jpg"), right_rect)
    cv2.imwrite(os.path.join(save_dir, f"frame_{idx:06d}_stitched.jpg"), stitched)

    print("已保存调试图:")
    print(f"  {os.path.join(save_dir, f'frame_{idx:06d}_stitched.jpg')}")


# ============================================================
# 8. 主实时循环
# ============================================================

def run_realtime(args: argparse.Namespace) -> None:
    """
    主实时流程。
    """
    maps = load_rectify_maps(args.map_file)
    stitch_params = load_stitch_params(args.stitch_param)

    if args.runtime_blend_width > 0:
        old_blend = stitch_params.blend_width
        stitch_params.blend_width = int(args.runtime_blend_width)
        print(f"[实时参数] blend_width: {old_blend} -> {stitch_params.blend_width}")
        print(f"[实时参数] seam_x: {args.runtime_seam_x}")
        print(f"[实时参数] right_x_shift: {args.runtime_right_x_shift}")
        print(f"[实时参数] right_y_shift: {args.runtime_right_y_shift}")
    
    # 检查摄像头输入分辨率是否和 map 文件一致。
    map_w, map_h = maps.raw_image_size

    if (args.width, args.height) != maps.raw_image_size:
        print("\n[警告] 你设置的摄像头分辨率和 map raw_image_size 不一致。")
        print(f"  args size : {(args.width, args.height)}")
        print(f"  map size  : {maps.raw_image_size}")
        print("  建议使用和标定完全一致的分辨率。")

    cap_left = open_usb_camera(
        args.left_device,
        args.width,
        args.height,
        args.fps,
        use_mjpg=not args.no_mjpg,
    )

    cap_right = open_usb_camera(
        args.right_device,
        args.width,
        args.height,
        args.fps,
        use_mjpg=not args.no_mjpg,
    )

    left_thread_cam = None
    right_thread_cam = None

    if args.threaded_capture:
        print("\n启用后台采集线程 threaded_capture=True")
        left_thread_cam = LatestFrameCamera(cap_left, name="left").start()
        right_thread_cam = LatestFrameCamera(cap_right, name="right").start()

        # 等待第一帧到来
        for _ in range(100):
            ok_l, _ = left_thread_cam.read_latest()
            ok_r, _ = right_thread_cam.read_latest()
            if ok_l and ok_r:
                break
            time.sleep(0.01)


    ensure_dir(args.save_dir)

    fps_counter = FPSCounter()

    frame_idx = 0
    save_idx = 0

    show_lines = args.show_lines
    show_rectified = False
    enable_color_balance = not args.disable_color_balance

    last_gain = None
    # 性能统计
    prof_count = 0
    prof_grab_ms = 0.0
    prof_proc_ms = 0.0
    prof_disp_ms = 0.0
    prof_total_ms = 0.0

    print("\n开始实时双目矫正 + 拼接")
    print("按键:")
    print("  q / Esc : 退出")
    print("  s       : 保存当前帧调试图")
    print("  b       : 开启/关闭亮度颜色补偿")
    print("  l       : 开启/关闭水平参考线")
    print("  r       : 开启/关闭左右 rectified 调试窗口")
    print("")
    print(f"color_balance: {enable_color_balance}")

    try:
        while True:
            t_loop0 = time.perf_counter()

            t_grab0 = time.perf_counter()

            if args.threaded_capture:
                ok, left_raw, right_raw = grab_pair_latest(
                    left_thread_cam,
                    right_thread_cam,
                )
            else:
                ok, left_raw, right_raw = grab_pair(
                    cap_left,
                    cap_right,
                    use_grab_retrieve=not args.no_grab_retrieve,
                )

            t_grab1 = time.perf_counter()

            if not ok:
                print("[警告] 摄像头读取失败")
                time.sleep(0.01)
                continue

            t_proc0 = time.perf_counter()

            # ----------------------------------------------------
            # raw -> 1920x1080 view
            # 注意：这里不再生成完整 left_rect / right_rect
            # ----------------------------------------------------
            try:
                crop_x = args.crop_x
                crop_y = args.crop_y

                if crop_x < 0:
                    crop_x = (stitch_params.output_width - args.view_width) // 2

                if crop_y < 0:
                    crop_y = (stitch_params.output_height - args.view_height) // 2

                stitched = stitch_view_roi_remap(
                    left_raw,
                    right_raw,
                    maps,
                    stitch_params,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    view_w=args.view_width,
                    view_h=args.view_height,
                    seam_x=args.runtime_seam_x,
                    right_x_shift=args.runtime_right_x_shift,
                    right_y_shift=args.runtime_right_y_shift,
                )
                last_gain = np.ones(3, dtype=np.float32)

            except RuntimeError as e:
                print(f"[错误] ROI remap 视口生成失败: {e}")
                break
            t_proc1 = time.perf_counter()

            fps = fps_counter.update()

            t_disp0 = time.perf_counter()
            # ----------------------------------------------------
            # 显示状态
            # ----------------------------------------------------
            display = stitched

            if show_lines:
                display = draw_horizontal_lines(display, args.epiline_step)

            status = (
                f"FPS:{fps:.1f} | "
                f"overlap:{stitch_params.overlap_px} | "
                f"dy:{stitch_params.vertical_offset} | "
                f"blend:{stitch_params.blend_width} | "
                f"balance:{'ON' if enable_color_balance else 'OFF'}"
            )

            display_with_text = display.copy()
            cv2.putText(
                display_with_text,
                status,
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            if not args.headless:
                show_img = resize_for_display(display_with_text, args.display_scale)
                cv2.imshow("realtime stitched", show_img)

                if show_rectified:
                    rect_wide = np.hstack([
                        left_rect[:min(left_rect.shape[0], right_rect.shape[0])],
                        right_rect[:min(left_rect.shape[0], right_rect.shape[0])],
                    ])
                    if show_lines:
                        rect_wide = draw_horizontal_lines(rect_wide, args.epiline_step)

                    cv2.imshow(
                        "left/right rectified",
                        resize_for_display(rect_wide, args.display_scale),
                    )

                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            # 额外支持终端输入按键：
            #   s 回车：保存调试图
            #   q 回车：退出
            #   b/l/r 回车：切换对应功能
            term_key = read_terminal_key()
            if term_key != 255:
                key = term_key

            t_disp1 = time.perf_counter()
            # ----------------------------------------------------
            # 按键处理
            # ----------------------------------------------------
            if key in (ord("q"), 27):
                print("退出实时拼接。")
                break

            elif key == ord("s"):
                ensure_dir(args.save_dir)

                cv2.imwrite(os.path.join(args.save_dir, f"frame_{save_idx:06d}_left_raw.jpg"), left_raw)
                cv2.imwrite(os.path.join(args.save_dir, f"frame_{save_idx:06d}_right_raw.jpg"), right_raw)
                cv2.imwrite(os.path.join(args.save_dir, f"frame_{save_idx:06d}_view_1920.jpg"), stitched)

                print("已保存:")
                print(os.path.join(args.save_dir, f"frame_{save_idx:06d}_view_1920.jpg"))

                save_idx += 1

            elif key == ord("b"):
                enable_color_balance = not enable_color_balance
                last_gain = None
                print(f"color_balance: {enable_color_balance}")

            elif key == ord("l"):
                show_lines = not show_lines
                print(f"show_lines: {show_lines}")

            elif key == ord("r"):
                print("ROI remap 模式下不再显示完整 left/right rectified。")
            # ----------------------------------------------------
            # 可选：周期性保存拼接图
            # ----------------------------------------------------
            if args.save_every > 0 and frame_idx % args.save_every == 0:
                path = os.path.join(args.save_dir, f"auto_stitched_{frame_idx:06d}.jpg")
                cv2.imwrite(path, stitched)

            t_loop1 = time.perf_counter()

            prof_count += 1
            prof_grab_ms += (t_grab1 - t_grab0) * 1000.0
            prof_proc_ms += (t_proc1 - t_proc0) * 1000.0
            prof_disp_ms += (t_disp1 - t_disp0) * 1000.0
            prof_total_ms += (t_loop1 - t_loop0) * 1000.0

            if prof_count >= 60:
                avg_grab = prof_grab_ms / prof_count
                avg_proc = prof_proc_ms / prof_count
                avg_disp = prof_disp_ms / prof_count
                avg_total = prof_total_ms / prof_count
                avg_fps = 1000.0 / max(avg_total, 1e-6)

                print(
                    f"[PROFILE] "
                    f"fps={avg_fps:.1f} | "
                    f"grab={avg_grab:.1f}ms | "
                    f"roi_remap={avg_proc:.1f}ms | "
                    f"display={avg_disp:.1f}ms | "
                    f"total={avg_total:.1f}ms"
                )

                prof_count = 0
                prof_grab_ms = 0.0
                prof_proc_ms = 0.0
                prof_disp_ms = 0.0
                prof_total_ms = 0.0

            frame_idx += 1
    finally:
        if left_thread_cam is not None:
            left_thread_cam.stop()

        if right_thread_cam is not None:
            right_thread_cam.stop()

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()




# ============================================================
# 9. 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime stereo rectification and stitching."
    )

    # 摄像头参数
    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="摄像头采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="禁用 MJPG，默认使用 MJPG")
    parser.add_argument(
        "--no-grab-retrieve",
        action="store_true",
        help="禁用 grab/retrieve 同步读取，改用连续 read",
    )
    parser.add_argument(
        "--threaded-capture",
        action="store_true",
        help="使用后台线程持续采集最新帧，减少主循环等待摄像头的时间",
    )

    # 离线文件
    parser.add_argument("--map-file", default=DEFAULT_MAP_FILE, help="双目 remap npz 文件")
    parser.add_argument("--stitch-param", default=DEFAULT_STITCH_PARAM, help="离线拼接参数 npz 文件")

    # 显示与保存
    parser.add_argument("--display-scale", type=float, default=0.35, help="显示缩放比例")
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="调试图保存目录")
    parser.add_argument("--save-every", type=int, default=0, help="每隔 N 帧自动保存 stitched，0 表示不自动保存")
    parser.add_argument("--view-width", type=int, default=1920, help="最终输出主画面宽度")
    parser.add_argument("--view-height", type=int, default=1080, help="最终输出主画面高度")
    parser.add_argument("--crop-x", type=int, default=-1, help="虚拟宽图裁剪起点 x，-1 表示居中")
    parser.add_argument("--crop-y", type=int, default=-1, help="虚拟宽图裁剪起点 y，-1 表示居中")
    
    parser.add_argument(
    "--runtime-blend-width",
    type=int,
    default=0,
    help="实时覆盖 blend_width。0 表示使用 stitch_params.npz 里的 blend_width",
    )
    parser.add_argument(
        "--runtime-seam-x",
        type=int,
        default=-1,
        help="实时接缝中心在 overlap 内的位置，-1 表示 overlap 中间",
    )
    parser.add_argument(
    "--runtime-right-x-shift",
    type=int,
    default=0,
    help="实时微调右图 rectified 取样 x 坐标，正数表示右图取样区域向右移动",
    )
    parser.add_argument(
    "--runtime-right-y-shift",
    type=int,
    default=0,
    help="实时微调右图 rectified 取样 y 坐标，正数表示右图取样区域向下移动",
)
    
    # 调试显示
    parser.add_argument("--show-lines", action="store_true", help="启动时显示水平参考线")
    parser.add_argument("--show-rectified", action="store_true", help="启动时显示左右 rectified 调试窗口")
    parser.add_argument("--epiline-step", type=int, default=80, help="水平参考线间隔")

    # 亮度颜色补偿
    parser.add_argument(
        "--disable-color-balance",
        action="store_true",
        help="关闭重叠区域 BGR 增益补偿",
    )
    parser.add_argument(
        "--gain-smooth",
        type=float,
        default=0.90,
        help="颜色补偿增益平滑系数。越接近 1 越稳定，越不容易闪烁。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ 实时双目矫正 + 拼接 ================")
    print(f"left_device       : {args.left_device}")
    print(f"right_device      : {args.right_device}")
    print(f"camera size       : {args.width} x {args.height}")
    print(f"fps               : {args.fps}")
    print(f"map_file          : {args.map_file}")
    print(f"stitch_param      : {args.stitch_param}")
    print(f"display_scale     : {args.display_scale}")
    print(f"use_mjpg          : {not args.no_mjpg}")
    print(f"use_grab_retrieve : {not args.no_grab_retrieve}")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n[提示] 当前没有检测到 DISPLAY。")
        print("如果你是通过 SSH 转发图形界面，请确认已经配置 DISPLAY。")
        print("例如：export DISPLAY=:0")

    run_realtime(args)


if __name__ == "__main__":
    main()
