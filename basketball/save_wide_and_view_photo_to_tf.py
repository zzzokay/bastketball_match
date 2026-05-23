#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
save_wide_stitch_photo_to_tf.py

用途：
    在 RK3588 / Linux / ELF2 上，从左右两个 USB 摄像头采集 1920x1080 图像，
    使用已经离线生成好的 stereo_rectify_maps_wide_good.npz 和 stitch_params_good.npz，
    生成完整宽幅拼接图，同时从宽图中裁出 1920x1080 主视口，并一起保存为 JPG 到 TF 卡目录。

适用场景：
    1. 想同时保存当前双目拼接后的完整宽图和 1920x1080 裁切窗口。
    2. 想验证当前 map / stitch 参数 / seam / right_x_shift / right_y_shift 的实际效果。
    3. 想把宽幅图保存到 TF 卡，方便拷贝到电脑上查看。

和实时脚本的区别：
    realtime_stereo_stitch_1920_view.py：
        主要用于实时 1920x1080 运镜输出，追求实时帧率。

    save_wide_stitch_photo_to_tf.py：
        主要用于拍照保存完整宽幅图和 1920x1080 主视口图，优先保证保存结果完整、清晰、方便调试。

典型用法：
python3 save_wide_and_view_photo_to_tf.py \
    --left-device /dev/video41 \
    --right-device /dev/video43 \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
    --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
    --save-dir /media/elf/7AC8-E830/basketball_photos \
    --runtime-seam-x 150 \
    --runtime-blend-width 40 \
    --runtime-right-x-shift 30 \
    --runtime-right-y-shift -5 \
    --view-width 1920 \
    --view-height 1080 \
    --crop-x -1 \
    --crop-y -1 \
    --jpg-quality 95 \
    --count 1

    默认会同时保存两张 JPG：
        wide_xxxx.jpg      ：完整宽幅图
        view1920_xxxx.jpg  ：从完整宽幅图裁出的 1920x1080 图

如果不知道 TF 卡挂载路径，先查看：
    lsblk
    df -h
    ls /media/elf/
    ls /media/elf/7AC8-E830

常见 TF 卡路径示例：
    /media/elf/XXXX-XXXX/
    /run/media/elf/XXXX-XXXX/
    /mnt/tfcard/

按键模式：
    默认不是实时按键保存，而是运行后自动保存 count 张并退出。
    如果你想预览后按 s 保存，可以加：
        --interactive

交互模式按键：
    s：同时保存一张完整宽幅 JPG 和一张 1920x1080 JPG
    q / Esc：退出
"""

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

# RK3588 / Mali 平台上，OpenCV 有时会尝试启用 OpenCL，
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
# 1. 默认参数
# ============================================================

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"

DEFAULT_MAP_FILE = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz"
DEFAULT_STITCH_PARAM = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz"

# 注意：这里无法提前知道你的 TF 卡挂载名，所以默认保存到当前工程目录。
# 真正使用时建议通过 --save-dir 指向 TF 卡目录，例如：
#   --save-dir /media/elf/TF_CARD/basketball_wide_photos
DEFAULT_SAVE_DIR = "/home/elf/work/basketball/tf_wide_photos"

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class RectifyMaps:
    """
    raw -> rectified 的 remap 查找表。

    raw_image_size:
        摄像头原始输入尺寸，例如 (1920, 1080)。

    rectified_size:
        remap 后的 rectified 图尺寸，例如 (2208, 1242)。

    left_map1 / left_map2:
        左摄 raw -> rectified 的 map。

    right_map1 / right_map2:
        右摄 raw -> rectified 的 map。
    """

    raw_image_size: Tuple[int, int]
    rectified_size: Tuple[int, int]
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


@dataclass
class StitchParams:
    """
    离线拼接参数。

    这些参数来自 stitch_params_good.npz。
    本脚本会在运行时允许覆盖：
        blend_width
        seam_x
        right_x_shift
        right_y_shift
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


# ============================================================
# 3. 基础工具函数
# ============================================================

def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    """
    从 npz 字段中读取图像尺寸。

    支持格式：
        [width, height]
        [[width, height]]
        numpy int 类型
    """
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 格式不正确: {value}")
    return int(flat[0]), int(flat[1])


def get_npz_int(data: np.lib.npyio.NpzFile, key: str, default: Optional[int] = None) -> int:
    """
    从 npz 中读取 int。

    如果字段不存在并且提供了 default，就返回 default。
    如果字段不存在且 default=None，就报错。
    """
    if key not in data.files:
        if default is None:
            raise RuntimeError(f"npz 缺少字段: {key}")
        return int(default)
    return int(np.array(data[key]).reshape(-1)[0])


def convert_maps_for_fast_remap(map1: np.ndarray, map2: np.ndarray):
    """
    把 float32 remap map 转成 fixed-point map。

    好处：
        cv2.remap 在 fixed-point map 下通常更快。

    注意：
        转换后 map1 通常是 int16，map2 通常是 uint16。
        cv2.remap 可以直接使用这种格式。
    """
    if map1.dtype == np.int16 and map2.dtype == np.uint16:
        return map1, map2

    return cv2.convertMaps(
        map1.astype(np.float32),
        map2.astype(np.float32),
        cv2.CV_16SC2,
    )


def current_timestamp() -> str:
    """生成适合文件名使用的时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def check_save_dir_writable(save_dir: str) -> None:
    """
    检查保存目录是否可写。

    对 TF 卡很重要：
        有时 TF 卡挂载为只读，或者路径写错。
        这里会写一个临时文件测试。
    """
    ensure_dir(save_dir)

    test_file = os.path.join(save_dir, ".write_test.tmp")
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise RuntimeError(f"保存目录不可写: {save_dir}\n原因: {e}")


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    """缩小图像用于 OpenCV 窗口预览。"""
    if abs(scale - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


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

    params = StitchParams(
        overlap_px=get_npz_int(data, "overlap_px"),
        vertical_offset=get_npz_int(data, "vertical_offset"),
        blend_width=get_npz_int(data, "blend_width", default=get_npz_int(data, "overlap_px")),

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
    """
    使用 V4L2 打开 USB 摄像头。

    默认使用 MJPG：
        双路 1920x1080@30fps 使用 YUYV 时 USB 带宽压力很大，
        MJPG 通常更容易跑满采集帧率。
    """
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

    if (real_w, real_h) != (width, height):
        print("[警告] 实际分辨率和请求分辨率不同。")
        print("       建议摄像头输出尺寸与 map 的 raw_image_size 完全一致。")

    return cap


def grab_pair(cap_left: cv2.VideoCapture, cap_right: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    采集一对左右图。

    这里使用 grab + retrieve：
        先对左右摄像头都 grab，再 retrieve 解码，
        可以减少左右帧的时间差。
    """
    ok_l = cap_left.grab()
    ok_r = cap_right.grab()

    if not ok_l or not ok_r:
        return False, None, None

    ret_l, frame_l = cap_left.retrieve()
    ret_r, frame_r = cap_right.retrieve()

    if not ret_l or not ret_r or frame_l is None or frame_r is None:
        return False, None, None

    return True, frame_l, frame_r


def warmup_cameras(cap_left: cv2.VideoCapture, cap_right: cv2.VideoCapture, frames: int) -> None:
    """
    丢弃前几帧，让自动曝光、自动白平衡更稳定。
    """
    frames = max(0, int(frames))
    if frames <= 0:
        return

    print(f"\n预热摄像头，丢弃前 {frames} 帧...")
    for _ in range(frames):
        grab_pair(cap_left, cap_right)
        time.sleep(0.005)


# ============================================================
# 6. ROI remap 与完整宽幅拼接
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

    为什么不直接切 map 后 remap？
        因为 runtime_right_x_shift / runtime_right_y_shift 可能让 ROI 超出 map 边界，
        例如 right_y_shift=-5 时，rect_y1 可能为 -5。

    本函数会：
        1. 创建一个固定大小的黑色输出图。
        2. 只对有效范围内的 map 做 cv2.remap。
        3. 把有效区域贴回固定大小输出图。

    这样即使 ROI 越界，也不会因为尺寸不一致导致程序崩溃。
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
    """
    从左右 raw 图直接生成完整宽幅拼接图。

    输出尺寸来自 stitch_params：
        params.output_width x params.output_height

    拼接结构：
        [左图非重叠区域] [overlap 区域] [右图非重叠区域]

    overlap 处理：
        overlap 不是整块都 alpha 融合。
        只有 seam_x 附近 blend_width 宽度做融合。
        这样可以减少近景重影。

    right_x_shift / right_y_shift：
        对右图 rectified 取样坐标做微调。
        你当前可接受参数是：
            right_x_shift = 30
            right_y_shift = -5
    """
    # 如果摄像头实际输出尺寸和 map 需要的 raw_image_size 不一致，进行兜底 resize。
    # 最好不要依赖这里，实际运行应保证摄像头输出和标定尺寸一致。
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

    # 左右图在 rectified 坐标中的 y 范围。
    left_y1 = params.left_y1
    left_y2 = params.left_y1 + output_h

    right_y1 = params.right_y1 + right_y_shift
    right_y2 = right_y1 + output_h

    # --------------------------------------------------------
    # 1. 左侧非重叠区域：直接取左图
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 2. overlap 区域：只在 seam 附近小范围融合
    # --------------------------------------------------------
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

    # overlap 左侧：直接取左图，避免大面积半透明重影。
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

    # overlap 右侧：直接取右图。
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

    # seam 附近：左右 alpha 融合。
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

    blend = (
        left_blend.astype(np.float32) * (1.0 - alpha)
        + right_blend.astype(np.float32) * alpha
    )
    blend = np.clip(blend, 0, 255).astype(np.uint8)

    wide[:, overlap_start + blend_x1:overlap_start + blend_x2] = blend

    # --------------------------------------------------------
    # 3. 右侧非重叠区域：直接取右图
    # --------------------------------------------------------
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
# 7. 保存 JPG
# ============================================================

def save_jpg(img: np.ndarray, save_dir: str, prefix: str, quality: int, timestamp: Optional[str] = None) -> str:
    """
    保存 JPG 文件。

    参数：
        img:
            要保存的 BGR 图像。

        save_dir:
            保存目录。建议指向 TF 卡挂载目录。

        prefix:
            文件名前缀，例如 wide_0000 或 view1920_0000。

        quality：
            1~100，越大质量越好、文件越大。
            建议 90~95。

        timestamp:
            可选时间戳。
            同一组 wide / view1920 使用同一个 timestamp，方便配对查看。
    """
    ensure_dir(save_dir)
    quality = int(np.clip(quality, 1, 100))

    if timestamp is None:
        timestamp = current_timestamp()

    filename = f"{prefix}_{timestamp}.jpg"
    path = os.path.join(save_dir, filename)

    ok = cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"保存 JPG 失败: {path}")

    return path


def crop_view_from_wide(
    wide: np.ndarray,
    crop_x: int,
    crop_y: int,
    view_w: int = 1920,
    view_h: int = 1080,
) -> Tuple[np.ndarray, int, int]:
    """
    从完整宽幅图中裁出一个固定大小的主视口图。

    为什么要从 wide 中裁？
        这个拍照脚本不是实时处理脚本，性能压力小。
        先生成完整宽图，再裁 1920x1080，可以保证两张保存图来自同一帧、同一套拼接结果，
        方便后面检查“宽图上的裁切位置”和“最终 1920x1080 输出”。

    crop_x / crop_y：
        裁切起点，坐标位于完整宽幅图坐标系。
        如果传入 -1，表示自动居中裁切。

    返回：
        view:
            裁出来的 1920x1080 图。

        used_crop_x / used_crop_y:
            实际使用的裁切起点。
            之后你做自动运镜时，可以把这里替换成运镜算法输出的位置。
    """
    wide_h, wide_w = wide.shape[:2]

    view_w = int(view_w)
    view_h = int(view_h)

    if view_w <= 0 or view_h <= 0:
        raise ValueError(f"view 尺寸不合法: {view_w}x{view_h}")

    if crop_x < 0:
        crop_x = (wide_w - view_w) // 2
    if crop_y < 0:
        crop_y = (wide_h - view_h) // 2

    crop_x = int(np.clip(crop_x, 0, max(0, wide_w - view_w)))
    crop_y = int(np.clip(crop_y, 0, max(0, wide_h - view_h)))

    # 正常情况下 wide 比 1920x1080 大，直接裁切即可。
    # 如果以后参数变化导致 wide 小于 view，这里会用黑边补齐，不让程序崩。
    view = np.zeros((view_h, view_w, 3), dtype=np.uint8)

    src_x2 = min(wide_w, crop_x + view_w)
    src_y2 = min(wide_h, crop_y + view_h)

    valid_w = src_x2 - crop_x
    valid_h = src_y2 - crop_y

    if valid_w > 0 and valid_h > 0:
        view[:valid_h, :valid_w] = wide[crop_y:src_y2, crop_x:src_x2]

    return view, crop_x, crop_y


def save_wide_and_view_jpg(
    wide: np.ndarray,
    save_dir: str,
    index: int,
    quality: int,
    crop_x: int,
    crop_y: int,
    view_w: int,
    view_h: int,
) -> Tuple[str, str, np.ndarray, int, int]:
    """
    同时保存完整宽幅图和 1920x1080 主视口图。

    保存结果示例：
        wide_0000_20260523_123456_789.jpg
        view1920_0000_20260523_123456_789.jpg

    两张图使用同一个时间戳，便于配对。
    """
    ts = current_timestamp()

    view, used_crop_x, used_crop_y = crop_view_from_wide(
        wide=wide,
        crop_x=crop_x,
        crop_y=crop_y,
        view_w=view_w,
        view_h=view_h,
    )

    wide_path = save_jpg(
        img=wide,
        save_dir=save_dir,
        prefix=f"wide_{index:04d}",
        quality=quality,
        timestamp=ts,
    )

    view_path = save_jpg(
        img=view,
        save_dir=save_dir,
        prefix=f"view1920_{index:04d}",
        quality=quality,
        timestamp=ts,
    )

    return wide_path, view_path, view, used_crop_x, used_crop_y


# ============================================================
# 8. 主流程
# ============================================================

def capture_and_save_once(
    cap_left: cv2.VideoCapture,
    cap_right: cv2.VideoCapture,
    maps: RectifyMaps,
    params: StitchParams,
    args: argparse.Namespace,
    index: int,
) -> Optional[Tuple[str, str]]:
    """
    采集一对左右图，拼接成完整宽幅图，
    同时保存：
        1. 完整宽幅 JPG
        2. 1920x1080 主视口 JPG
    """
    ok, left_raw, right_raw = grab_pair(cap_left, cap_right)
    if not ok:
        print("[警告] 摄像头采集失败，本次不保存。")
        return None

    t0 = time.perf_counter()

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

    t1 = time.perf_counter()

    wide_path, view_path, view, used_crop_x, used_crop_y = save_wide_and_view_jpg(
        wide=wide,
        save_dir=args.save_dir,
        index=index,
        quality=args.jpg_quality,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        view_w=args.view_width,
        view_h=args.view_height,
    )

    t2 = time.perf_counter()

    print("\n已同时保存 JPG:")
    print(f"  wide      : {wide_path}")
    print(f"  view1920  : {view_path}")
    print(f"  wide size : {wide.shape[1]} x {wide.shape[0]}")
    print(f"  view size : {view.shape[1]} x {view.shape[0]}")
    print(f"  crop_xy   : ({used_crop_x}, {used_crop_y})")
    print(f"  stitch time       : {(t1 - t0) * 1000.0:.1f} ms")
    print(f"  save both jpg time: {(t2 - t1) * 1000.0:.1f} ms")

    return wide_path, view_path


def run_auto_save(args: argparse.Namespace) -> None:
    """
    自动保存 count 张后退出。
    """
    maps = load_rectify_maps(args.map_file)
    params = load_stitch_params(args.stitch_param)

    check_save_dir_writable(args.save_dir)

    print("\n运行时拼接参数:")
    print(f"  runtime_seam_x       : {args.runtime_seam_x}")
    print(f"  runtime_blend_width  : {args.runtime_blend_width}")
    print(f"  runtime_right_x_shift: {args.runtime_right_x_shift}")
    print(f"  runtime_right_y_shift: {args.runtime_right_y_shift}")
    print(f"  save_dir             : {args.save_dir}")
    print(f"  jpg_quality          : {args.jpg_quality}")
    print(f"  view_size            : {args.view_width} x {args.view_height}")
    print(f"  crop_x / crop_y      : {args.crop_x}, {args.crop_y}")
    print(f"  view_size            : {args.view_width} x {args.view_height}")
    print(f"  crop_x / crop_y      : {args.crop_x}, {args.crop_y}")

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

    try:
        if (args.width, args.height) != maps.raw_image_size:
            print("\n[警告] 摄像头输入尺寸和 map 的 raw_image_size 不一致。")
            print(f"  camera size : {(args.width, args.height)}")
            print(f"  map size    : {maps.raw_image_size}")
            print("  这会触发 resize，可能影响清晰度和拼接精度。")

        warmup_cameras(cap_left, cap_right, args.warmup_frames)

        for i in range(args.count):
            capture_and_save_once(cap_left, cap_right, maps, params, args, i)

            if i != args.count - 1:
                time.sleep(max(0.0, args.interval))

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


def run_interactive(args: argparse.Namespace) -> None:
    """
    交互模式：
        显示缩小后的完整宽图预览。
        按 s 保存完整宽图 JPG。
        按 q / Esc 退出。

    注意：
        这个脚本主要用于保存照片，不追求最高实时帧率。
    """
    maps = load_rectify_maps(args.map_file)
    params = load_stitch_params(args.stitch_param)

    check_save_dir_writable(args.save_dir)

    print("\n运行时拼接参数:")
    print(f"  runtime_seam_x       : {args.runtime_seam_x}")
    print(f"  runtime_blend_width  : {args.runtime_blend_width}")
    print(f"  runtime_right_x_shift: {args.runtime_right_x_shift}")
    print(f"  runtime_right_y_shift: {args.runtime_right_y_shift}")
    print(f"  save_dir             : {args.save_dir}")
    print(f"  jpg_quality          : {args.jpg_quality}")

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

    save_idx = 0

    try:
        warmup_cameras(cap_left, cap_right, args.warmup_frames)

        print("\n进入交互模式：")
        print("  s       : 同时保存当前完整宽幅图和 1920x1080 主视口图")
        print("  q / Esc : 退出")

        while True:
            ok, left_raw, right_raw = grab_pair(cap_left, cap_right)
            if not ok:
                print("[警告] 摄像头采集失败")
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

            preview = resize_for_display(wide, args.display_scale)
            cv2.putText(
                preview,
                "s: save wide + 1920 view | q: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("wide stitched preview", preview)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("退出。")
                break

            if key == ord("s"):
                wide_path, view_path, view, used_crop_x, used_crop_y = save_wide_and_view_jpg(
                    wide=wide,
                    save_dir=args.save_dir,
                    index=save_idx,
                    quality=args.jpg_quality,
                    crop_x=args.crop_x,
                    crop_y=args.crop_y,
                    view_w=args.view_width,
                    view_h=args.view_height,
                )
                print("已同时保存:")
                print(f"  wide     : {wide_path}")
                print(f"  view1920 : {view_path}")
                print(f"  crop_xy  : ({used_crop_x}, {used_crop_y})")
                save_idx += 1

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


# ============================================================
# 9. 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture stereo cameras, save full wide JPG and 1920x1080 view JPG to TF card."
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

    # 保存参数
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="JPG 保存目录，建议指向 TF 卡挂载目录")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPG 质量，1~100，建议 90~95")
    parser.add_argument("--count", type=int, default=1, help="自动保存张数")
    parser.add_argument("--interval", type=float, default=1.0, help="自动保存多张时，两张之间的间隔秒数")
    parser.add_argument("--warmup-frames", type=int, default=20, help="保存前丢弃多少帧，让曝光/白平衡稳定")

    # 1920x1080 主视口保存参数
    parser.add_argument("--view-width", type=int, default=1920, help="同时保存的主视口图宽度")
    parser.add_argument("--view-height", type=int, default=1080, help="同时保存的主视口图高度")
    parser.add_argument("--crop-x", type=int, default=-1, help="主视口在宽幅图中的裁切起点 x，-1 表示居中")
    parser.add_argument("--crop-y", type=int, default=-1, help="主视口在宽幅图中的裁切起点 y，-1 表示居中")

    # 拼接微调参数：默认使用你当前验证过可接受的参数
    parser.add_argument("--runtime-seam-x", type=int, default=150, help="接缝中心在 overlap 内的位置")
    parser.add_argument("--runtime-blend-width", type=int, default=40, help="实时融合宽度，重影明显时不要太大")
    parser.add_argument("--runtime-right-x-shift", type=int, default=30, help="右图 rectified 取样 x 微调")
    parser.add_argument("--runtime-right-y-shift", type=int, default=-5, help="右图 rectified 取样 y 微调")

    # 交互预览
    parser.add_argument("--interactive", action="store_true", help="进入预览窗口，按 s 保存，按 q 退出")
    parser.add_argument("--display-scale", type=float, default=0.25, help="交互预览窗口缩放比例")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ 保存宽幅 + 1920x1080 JPG ================")
    print(f"left_device       : {args.left_device}")
    print(f"right_device      : {args.right_device}")
    print(f"camera size       : {args.width} x {args.height}")
    print(f"fps               : {args.fps}")
    print(f"map_file          : {args.map_file}")
    print(f"stitch_param      : {args.stitch_param}")
    print(f"save_dir          : {args.save_dir}")
    print(f"use_mjpg          : {not args.no_mjpg}")

    if args.interactive:
        run_interactive(args)
    else:
        run_auto_save(args)


if __name__ == "__main__":
    main()
