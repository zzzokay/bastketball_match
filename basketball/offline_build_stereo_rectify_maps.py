#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
offline_build_stereo_rectify_maps/
├── images/
│   ├── Left/
│   └── Right/
├── stereo_rectify_maps_wide.npz
├── corners_000000.jpg
├── test_left_rect.jpg
├── test_right_rect.jpg
└── test_wide_lines.jpg


offline_build_stereo_rectify_maps.py

用途：
    离线阶段使用。
    负责从左右摄像头采集双目标定图片，并根据这些图片生成"原始图 -> 双目极线校正图"的 remap 查找表。

你当前选择的阶段流程对应为：

    原始左右图像
        ↓
    使用已有单目标定 npz 做单目畸变模型读取
        ↓
    采集双目标定图片时，先把画面单目畸变矫正后保存
        ↓
    用这些"已单目矫正"的左右图片做 stereoCalibrate
        ↓
    stereoRectify 生成 R1/R2/P1/P2/Q
        ↓
    生成最终可实时使用的 remap 查找表
        ↓
    保存到 stereo_rectify_maps_wide.npz

重要设计：
    1. 采集双目标定图时保存的是"已单目畸变矫正图"，这样更方便人工检查。
    2. 实时使用时保存的是"原始 raw 图直接 remap 到最终极线校正图"的查找表。
       这样实时阶段只做一次 cv2.remap，避免：
           raw -> 单目矫正 remap -> 双目极线矫正 remap
       这种两次插值造成画质下降和速度下降。
    3. 对两个摄像头夹角较大的情况，默认使用：
           rectify_alpha = 1.0
           zero_disparity = False
           out_scale = 1.15
       这样更偏向"保留视野、少裁切、少强行放大"，代价是可能有黑边。
       后续真正融合前可以再裁剪有效区域。

典型用法一：采集双目标定图
    python3 offline_build_stereo_rectify_maps.py \
        --mode capture \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
        --right-calib-file /home/elf/work/basketball/camera_calib.npz \
        --capture-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/images \
        --board-cols 11 \
        --board-rows 8 \
        --square-size 22 \
        --display-scale 0.25

典型用法二：根据采集好的双目标定图片生成 remap 查找表
    python3 offline_build_stereo_rectify_maps.py \
        --mode build \
        --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
        --right-calib-file /home/elf/work/basketball/camera_calib.npz \
        --capture-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/images \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
        --board-cols 11 \
        --board-rows 8 \
        --square-size 22 \
        --rectify-alpha 1.0 \
        --out-scale 1.15 \
        --headless

典型用法三：用一对图片测试生成好的 remap
    python3 offline_build_stereo_rectify_maps.py \
        --mode test-image \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
        --left-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/images/Left/left_000020.png \
        --right-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/images/Right/right_000020.png \
        --output-dir /home/elf/work/basketball/stereo_rectify_debug \
        --input-already-undistorted \
        --headless 

按键：
    capture 模式：
        s：保存当前左右"已单目畸变矫正"的双目标定图；capture 阶段默认不检测棋盘格
        q / Esc：退出
"""

import argparse
import glob
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

# RK3588 / Mali 平台上 OpenCV 有时会尝试加载 OpenCL 内核缓存，
# 可能出现 CL_INVALID_BINARY。这里直接关闭 OpenCL，避免无意义报错和额外开销。
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# ============================================================
# 1. 默认参数
# ============================================================

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"

DEFAULT_LEFT_CALIB_FILE = "/home/elf/work/basketball/camera_usb2_calib.npz"
DEFAULT_RIGHT_CALIB_FILE = "/home/elf/work/basketball/camera_calib.npz"

DEFAULT_CAPTURE_DIR = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/images"
DEFAULT_MAP_FILE = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz"
DEFAULT_OUTPUT_DIR = "/home/elf/work/basketball/offline_build_stereo_rectify_maps"

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


# ============================================================
# 2. 小工具
# ============================================================

class TerminalKeyReader:
    """
    非阻塞终端按键读取器。

    在 RK3588 / Linux 桌面环境里，OpenCV 窗口有时拿不到键盘焦点。
    这个类可以直接从终端读取 q/s，避免窗口焦点问题。
    """

    def __init__(self):
        self.enabled = False
        self.old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.enabled = True
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def read_key(self) -> Optional[str]:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1)
        return None


class FPSCounter:
    """简单 FPS 统计器。"""

    def __init__(self):
        self.last_time = time.time()
        self.count = 0
        self.fps = 0.0

    def update(self) -> float:
        self.count += 1
        now = time.time()
        dt = now - self.last_time
        if dt >= 1.0:
            self.fps = self.count / dt
            self.count = 0
            self.last_time = now
        return self.fps


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def read_key_from_cv_and_terminal(cv_key: int, terminal_key: Optional[str]) -> Optional[str]:
    """
    统一处理 cv2.waitKey 和终端按键。
    """
    if terminal_key is not None:
        return terminal_key
    if cv_key != -1:
        return chr(cv_key & 0xFF)
    return None


def open_usb_camera(device: str, width: int, height: int, fps: int, use_mjpg: bool = True) -> cv2.VideoCapture:
    """
    使用 V4L2 打开 USB 摄像头。

    注意：
        1080P@30fps 双路 USB 摄像头，如果使用 YUYV 未压缩格式，USB 带宽压力很大。
        所以默认尝试 MJPG。
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device}")

    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))

    print(f"摄像头已打开: {device}")
    print(f"  request size : {width}x{height}")
    print(f"  real size    : {real_w}x{real_h}")
    print(f"  fps          : {real_fps}")
    print(f"  fourcc       : {fourcc_str}")

    if (real_w, real_h) != (width, height):
        print("[警告] 实际采集分辨率与请求分辨率不同。标定和实时运行最好保持完全一致。")

    return cap


# ============================================================
# 3. 单目标定读取与单目畸变矫正 map
# ============================================================

@dataclass
class MonoCalib:
    """
    单目标定数据。

    K_raw / D_raw:
        原始相机内参和畸变系数。

    K_undist:
        单目畸变矫正后图像使用的新内参。
        采集双目标定图时保存的是 undistorted image，所以后续 stereoCalibrate 用 K_undist + 零畸变。

    map1 / map2:
        原始图 -> 单目畸变矫正图 的 remap 表。
    """

    calib_file: str
    image_size: Tuple[int, int]
    K_raw: np.ndarray
    D_raw: np.ndarray
    K_undist: np.ndarray
    D_undist: np.ndarray
    roi: Tuple[int, int, int, int]
    map1: np.ndarray
    map2: np.ndarray


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    """
    兼容不同 npz 中 image_size 的保存方式：
        [width, height]
        [[width, height]]
        numpy int 类型
    """
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 格式不正确: {value}")
    return int(flat[0]), int(flat[1])


def scale_dist_coeffs(dist_coeffs_raw: np.ndarray, dist_scale: float) -> np.ndarray:
    """
    按你之前单目脚本的习惯缩放径向畸变系数。

    OpenCV 常见畸变系数：
        [k1, k2, p1, p2, k3]

    这里只缩放 k1/k2/k3。
    p1/p2 是切向畸变，一般不建议随意缩放。
    """
    D = dist_coeffs_raw.astype(np.float64).copy()
    flat = D.reshape(-1)

    if flat.size >= 1:
        flat[0] *= dist_scale
    if flat.size >= 2:
        flat[1] *= dist_scale
    if flat.size >= 5:
        flat[4] *= dist_scale

    return flat.reshape(D.shape)


def load_mono_calib_and_create_maps(
    calib_file: str,
    runtime_image_size: Tuple[int, int],
    mono_alpha: float,
    mono_dist_scale: float,
) -> MonoCalib:
    """
    加载单目标定文件，生成原始图 -> 单目畸变矫正图的 remap 表。

    mono_alpha:
        cv2.getOptimalNewCameraMatrix 的 alpha。
        0：尽量裁黑边，可用画面更满，但视野少一些。
        1：尽量保留视野，可能有黑边。
        对后续大夹角拼接，一般建议从 0.0 或 0.5 试起。
        如果特别想保留视野，可以用 1.0。

    mono_dist_scale:
        保持和你生成单目矫正图时一致。
        你之前常用 0.8，则这里也建议 0.8。
    """
    if not os.path.exists(calib_file):
        raise RuntimeError(f"找不到单目标定文件: {calib_file}")

    data = np.load(calib_file)
    K_raw = data["camera_matrix"].astype(np.float64)
    D_raw_file = data["dist_coeffs"].astype(np.float64)

    if "image_size" in data.files:
        saved_size = parse_image_size(data["image_size"])
        if saved_size != runtime_image_size:
            print("[警告] 单目标定文件中的 image_size 与当前采集尺寸不一致。")
            print(f"  calib   : {saved_size}")
            print(f"  runtime : {runtime_image_size}")
            print("  强烈建议：单目标定、双目标定采集、实时运行使用同一分辨率。")

    D_raw = scale_dist_coeffs(D_raw_file, mono_dist_scale)

    K_undist, roi = cv2.getOptimalNewCameraMatrix(
        K_raw,
        D_raw,
        runtime_image_size,
        mono_alpha,
        runtime_image_size,
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K_raw,
        D_raw,
        None,
        K_undist,
        runtime_image_size,
        cv2.CV_16SC2,
    )

    print(f"单目 map 已创建: {calib_file}")
    print(f"  runtime_image_size : {runtime_image_size}")
    print(f"  mono_alpha         : {mono_alpha}")
    print(f"  mono_dist_scale    : {mono_dist_scale}")
    print(f"  roi                : {tuple(int(x) for x in roi)}")

    return MonoCalib(
        calib_file=calib_file,
        image_size=runtime_image_size,
        K_raw=K_raw,
        D_raw=D_raw,
        K_undist=K_undist,
        D_undist=np.zeros((5, 1), dtype=np.float64),
        roi=tuple(int(x) for x in roi),
        map1=map1,
        map2=map2,
    )


# ============================================================
# 4. 棋盘格角点检测
# ============================================================

def create_object_points(board_cols: int, board_rows: int, square_size: float) -> np.ndarray:
    """
    创建棋盘格三维点。

    board_cols / board_rows 是"内角点数量"，不是格子数量。
    例如你的 11 x 8 inner corners：
        board_cols = 11
        board_rows = 8

    输出坐标单位由 square_size 决定：
        square_size = 22 表示 22 mm。
    """
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size)
    return objp


def find_chessboard_corners(gray: np.ndarray, board_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
    """
    检测棋盘格角点，并做亚像素优化。

    优先使用 findChessboardCornersSB：
        OpenCV 新版本提供，稳定性通常更好。
    如果当前 OpenCV 没有这个函数，则回退到 findChessboardCorners。
    """
    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):
        flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE
        try:
            found, corners = cv2.findChessboardCornersSB(gray, board_size, flags_sb)
        except cv2.error:
            found, corners = False, None

    if not found:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, board_size, flags)

        if found and corners is not None:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return bool(found), corners


def draw_corners_debug(img: np.ndarray, board_size: Tuple[int, int], found: bool, corners: Optional[np.ndarray]) -> np.ndarray:
    debug = img.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(debug, board_size, corners, found)
    return debug


def corner_direction_score(corners_ref, corners_test, board_cols, board_rows):
    """
    比较两组棋盘格角点的方向是否一致。

    这里不比较绝对坐标，因为左右相机视角不同；
    只比较棋盘格横向方向和纵向方向是否一致。
    """
    ref = corners_ref.reshape(board_rows, board_cols, 2)
    tst = corners_test.reshape(board_rows, board_cols, 2)

    # 横向方向：每一行最后一个点 - 第一个点，然后取平均
    ref_x = np.mean(ref[:, -1, :] - ref[:, 0, :], axis=0)
    tst_x = np.mean(tst[:, -1, :] - tst[:, 0, :], axis=0)

    # 纵向方向：最后一行点 - 第一行点，然后取平均
    ref_y = np.mean(ref[-1, :, :] - ref[0, :, :], axis=0)
    tst_y = np.mean(tst[-1, :, :] - tst[0, :, :], axis=0)

    def norm(v):
        n = np.linalg.norm(v)
        if n < 1e-6:
            return v
        return v / n

    ref_x = norm(ref_x)
    tst_x = norm(tst_x)
    ref_y = norm(ref_y)
    tst_y = norm(tst_y)

    return float(np.dot(ref_x, tst_x) + np.dot(ref_y, tst_y))


def reorder_corners_like_left(corners_left, corners_right, board_cols, board_rows):
    """
    让右图角点顺序尽量和左图一致。

    OpenCV 棋盘格检测有时会出现 180 度翻转、左右翻转、上下翻转。
    对单目标定影响不一定明显，但对双目标定影响很大。
    """
    r = corners_right.reshape(board_rows, board_cols, 1, 2)

    candidates = [
        r,                    # 原始
        r[::-1, :, :, :],      # 上下翻转
        r[:, ::-1, :, :],      # 左右翻转
        r[::-1, ::-1, :, :],   # 180 度翻转
    ]

    best = None
    best_score = -1e9

    for cand in candidates:
        cand_flat = cand.reshape(-1, 1, 2)
        score = corner_direction_score(corners_left, cand_flat, board_cols, board_rows)
        if score > best_score:
            best_score = score
            best = cand_flat

    return best.astype(np.float32), best_score


def apply_right_corner_transform(
    corners_left: np.ndarray,
    corners_right: np.ndarray,
    board_cols: int,
    board_rows: int,
    mode: str,
) -> Tuple[np.ndarray, float]:
    """
    手动/自动选择右图角点顺序。

    为什么需要这个：
        棋盘格是规则图案，OpenCV 检测出的角点顺序有时和物理棋盘坐标不一致。
        order_score 接近 2 只说明图像方向相似，不保证物理角点对应正确。
        如果角点对应顺序错了，stereoCalibrate 会得到错误 R/T，
        stereoRectify 后可能出现右图倒置、强裁切、强拉伸。

    mode:
        auto_direction : 使用原来的方向相似度自动选择
        orig           : 不改变右图角点顺序
        flip_ud        : 上下翻转右图角点顺序
        flip_lr        : 左右翻转右图角点顺序
        rot180         : 右图角点顺序旋转 180 度
    """
    if mode == "auto_direction":
        return reorder_corners_like_left(corners_left, corners_right, board_cols, board_rows)

    r = corners_right.reshape(board_rows, board_cols, 1, 2)

    candidates = {
        "orig": r,
        "flip_ud": r[::-1, :, :, :],
        "flip_lr": r[:, ::-1, :, :],
        "rot180": r[::-1, ::-1, :, :],
    }

    if mode not in candidates:
        raise ValueError(f"未知 right_corner_transform: {mode}")

    out = candidates[mode].reshape(-1, 1, 2).astype(np.float32)
    score = corner_direction_score(corners_left, out, board_cols, board_rows)
    return out, score


# ============================================================
# 5. capture 模式：采集已单目畸变矫正的双目标定图
# ============================================================

def run_capture_mode(args: argparse.Namespace) -> None:
    """
    快速采集双目标定图。

    这个版本和之前最大的区别：
        1. 预览阶段不做棋盘格检测。
        2. 预览阶段默认不做单目 remap。
        3. 只有按下 s 保存时，才对当前 raw 图做单目畸变矫正并保存。
        4. build 模式再统一批量检测棋盘格，自动跳过不可用图片。

    这样 capture 阶段的循环只剩：
        read left/right
        resize 检查
        缩小预览
        waitKey / 终端按键

    保存目录：
        capture_dir/Left/left_000000.png
        capture_dir/Right/right_000000.png

    默认保存的是"已单目畸变矫正图"，后续 build 模式直接用这些图做 stereoCalibrate。
    """
    image_size = (args.width, args.height)
    board_size = (args.board_cols, args.board_rows)

    # 这里只是提前创建 map，后面按 s 保存时才使用 cv2.remap。
    # 不要在每一帧预览时都 remap，否则 1080P 双路会明显拖慢。
    left_calib = load_mono_calib_and_create_maps(
        args.left_calib_file,
        image_size,
        args.mono_alpha,
        args.mono_dist_scale,
    )
    right_calib = load_mono_calib_and_create_maps(
        args.right_calib_file,
        image_size,
        args.mono_alpha,
        args.mono_dist_scale,
    )

    left_dir = os.path.join(args.capture_dir, "Left")
    right_dir = os.path.join(args.capture_dir, "Right")
    ensure_dir(left_dir)
    ensure_dir(right_dir)

    if args.save_raw:
        raw_left_dir = os.path.join(args.capture_dir, "RawLeft")
        raw_right_dir = os.path.join(args.capture_dir, "RawRight")
        ensure_dir(raw_left_dir)
        ensure_dir(raw_right_dir)
    else:
        raw_left_dir = raw_right_dir = ""

    cap_left = open_usb_camera(args.left_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)
    cap_right = open_usb_camera(args.right_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)

    existing_left = sorted(
        glob.glob(os.path.join(left_dir, "*.png"))
        + glob.glob(os.path.join(left_dir, "*.jpg"))
        + glob.glob(os.path.join(left_dir, "*.jpeg"))
    )
    save_idx = len(existing_left)
    fps_counter = FPSCounter()
    frame_idx = 0

    print("\n开始快速采集双目标定图")
    print("  s：保存当前左右图；保存时才执行单目畸变矫正")
    print("  q/Esc：退出")
    print("  capture 阶段默认不检测棋盘格，build 阶段会自动检测并跳过不可用图片。")
    print("  如需在保存瞬间检测棋盘格，可加 --capture-detect-on-save。")
    print("  如需预览单目矫正后的画面，可加 --preview-undistorted，但 FPS 会降低。")

    with TerminalKeyReader() as key_reader:
        while True:
            ret_l, raw_left = cap_left.read()
            ret_r, raw_right = cap_right.read()

            if not ret_l or raw_left is None:
                print("[警告] 左相机读取失败")
                continue
            if not ret_r or raw_right is None:
                print("[警告] 右相机读取失败")
                continue

            if (raw_left.shape[1], raw_left.shape[0]) != image_size:
                raw_left = cv2.resize(raw_left, image_size)
            if (raw_right.shape[1], raw_right.shape[0]) != image_size:
                raw_right = cv2.resize(raw_right, image_size)

            fps = fps_counter.update()
            frame_idx += 1

            cv_key = -1
            if not args.headless and (frame_idx % max(1, args.display_every) == 0):
                if args.preview_undistorted:
                    # 仅调试用：这个会让 capture 阶段变慢。
                    preview_left = cv2.remap(raw_left, left_calib.map1, left_calib.map2, cv2.INTER_LINEAR)
                    preview_right = cv2.remap(raw_right, right_calib.map1, right_calib.map2, cv2.INTER_LINEAR)
                    preview_title = "undistorted preview"
                else:
                    # 快速预览：直接显示 raw 图，不做 remap、不检测棋盘格。
                    preview_left = raw_left
                    preview_right = raw_right
                    preview_title = "raw preview"

                # 先缩小单张图，再 hstack，避免先拼接 3840x1080 大图再缩放造成额外开销。
                if abs(args.display_scale - 1.0) > 1e-6:
                    small_left = resize_for_display(preview_left, args.display_scale)
                    small_right = resize_for_display(preview_right, args.display_scale)
                else:
                    small_left = preview_left
                    small_right = preview_right

                ph = min(small_left.shape[0], small_right.shape[0])
                wide = np.hstack([small_left[:ph], small_right[:ph]])

                status = (
                    f"FAST CAPTURE {preview_title} | FPS:{fps:.1f} | "
                    f"saved:{save_idx} | s save | q quit"
                )
                cv2.putText(
                    wide,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("fast stereo capture", wide)
                cv_key = cv2.waitKey(1)

            key = read_key_from_cv_and_terminal(cv_key, key_reader.read_key())

            if key is None:
                continue

            if key in ("q", "Q") or cv_key == 27:
                print("退出采集。")
                break

            if key in ("s", "S"):
                # 按下 s 时才做单目畸变矫正，保存给 build 模式使用。
                left_undist = cv2.remap(raw_left, left_calib.map1, left_calib.map2, cv2.INTER_LINEAR)
                right_undist = cv2.remap(raw_right, right_calib.map1, right_calib.map2, cv2.INTER_LINEAR)

                should_save = True

                # 默认不检测棋盘格；只有用户显式打开时，才在保存瞬间检测一次。
                if args.capture_detect_on_save:
                    left_gray = cv2.cvtColor(left_undist, cv2.COLOR_BGR2GRAY)
                    right_gray = cv2.cvtColor(right_undist, cv2.COLOR_BGR2GRAY)
                    found_l, _ = find_chessboard_corners(left_gray, board_size)
                    found_r, _ = find_chessboard_corners(right_gray, board_size)
                    print(f"保存前棋盘格检测: left={found_l}, right={found_r}")

                    if args.reject_without_board and not (found_l and found_r):
                        should_save = False
                        print("[未保存] 当前帧左右没有同时检测到棋盘格。")

                if should_save:
                    ext = args.save_ext.lower().lstrip(".")
                    if ext not in ("png", "jpg", "jpeg"):
                        ext = "png"

                    left_path = os.path.join(left_dir, f"left_{save_idx:06d}.{ext}")
                    right_path = os.path.join(right_dir, f"right_{save_idx:06d}.{ext}")

                    cv2.imwrite(left_path, left_undist)
                    cv2.imwrite(right_path, right_undist)

                    if args.save_raw:
                        raw_left_path = os.path.join(raw_left_dir, f"raw_left_{save_idx:06d}.{ext}")
                        raw_right_path = os.path.join(raw_right_dir, f"raw_right_{save_idx:06d}.{ext}")
                        cv2.imwrite(raw_left_path, raw_left)
                        cv2.imwrite(raw_right_path, raw_right)

                    print(f"已保存第 {save_idx} 组已单目矫正图片：")
                    print(f"  {left_path}")
                    print(f"  {right_path}")
                    if args.save_raw:
                        print("  同时已保存 raw 备份。")

                    save_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


# ============================================================
# 6. build 模式：从已采集图片生成双目参数和 remap
# ============================================================

def extract_index_from_name(path: str, prefix: str) -> Optional[str]:
    """
    从 left_000020.png / right_000020.png 中提取 000020。
    """
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)

    if not stem.startswith(prefix + "_"):
        return None

    return stem.split("_", 1)[1]


def list_pair_images(capture_dir: str) -> Tuple[List[str], List[str]]:
    """
    严格按编号配对 Left / Right 图片。

    好处：
        即使某一边缺图，也不会导致后面的图片整体错位。
    """
    left_dir = os.path.join(capture_dir, "Left")
    right_dir = os.path.join(capture_dir, "Right")

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

    left_map = {}
    right_map = {}

    for p in patterns:
        for f in glob.glob(os.path.join(left_dir, p)):
            idx = extract_index_from_name(f, "left")
            if idx is not None:
                left_map[idx] = f

        for f in glob.glob(os.path.join(right_dir, p)):
            idx = extract_index_from_name(f, "right")
            if idx is not None:
                right_map[idx] = f

    common_ids = sorted(set(left_map.keys()) & set(right_map.keys()))

    missing_left = sorted(set(right_map.keys()) - set(left_map.keys()))
    missing_right = sorted(set(left_map.keys()) - set(right_map.keys()))

    if missing_left:
        print("[警告] 这些编号缺少左图，将跳过:", missing_left)
    if missing_right:
        print("[警告] 这些编号缺少右图，将跳过:", missing_right)

    if not common_ids:
        raise RuntimeError(f"没有找到可配对的双目标定图片：{left_dir} / {right_dir}")

    left_files = [left_map[i] for i in common_ids]
    right_files = [right_map[i] for i in common_ids]

    print(f"找到可配对图片数量: {len(common_ids)}")
    print(f"编号范围: {common_ids[0]} -> {common_ids[-1]}")

    return left_files, right_files

def collect_stereo_corners(
    left_files: List[str],
    right_files: List[str],
    board_cols: int,
    board_rows: int,
    square_size: float,
    output_dir: str,
    save_debug: bool,
    right_corner_transform: str = "auto_direction",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """
    从左右已单目矫正图片中提取棋盘格角点。

    返回：
        object_points
        left_image_points
        right_image_points
        image_size
    """
    board_size = (board_cols, board_rows)
    objp = create_object_points(board_cols, board_rows, square_size)

    object_points: List[np.ndarray] = []
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []

    image_size: Optional[Tuple[int, int]] = None

    if save_debug:
        ensure_dir(output_dir)

    print("\n开始检测双目标定图片角点：")

    for idx, (lf, rf) in enumerate(zip(left_files, right_files)):
        left = cv2.imread(lf)
        right = cv2.imread(rf)

        if left is None:
            print(f"[跳过] 左图读取失败: {lf}")
            continue
        if right is None:
            print(f"[跳过] 右图读取失败: {rf}")
            continue

        if image_size is None:
            image_size = (left.shape[1], left.shape[0])

        if (left.shape[1], left.shape[0]) != image_size:
            print(f"[跳过] 左图尺寸不一致: {lf}")
            continue
        if (right.shape[1], right.shape[0]) != image_size:
            print(f"[跳过] 右图尺寸不一致: {rf}")
            continue

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_chessboard_corners(gray_l, board_size)
        found_r, corners_r = find_chessboard_corners(gray_r, board_size)

        if found_l and found_r and corners_l is not None and corners_r is not None:
            corners_l = corners_l.astype(np.float32)
            corners_r = corners_r.astype(np.float32)

            # 关键修复：
            # 右图角点顺序不一定和物理棋盘坐标一致。
            # 这里允许通过 --right-corner-transform 手动指定右图角点顺序。
            corners_r, order_score = apply_right_corner_transform(
                corners_l,
                corners_r,
                board_cols,
                board_rows,
                right_corner_transform,
            )

            object_points.append(objp.copy())
            left_points.append(corners_l)
            right_points.append(corners_r)

            print(
                f"[OK] {idx:03d}: {os.path.basename(lf)} / {os.path.basename(rf)} "
                f"order_score={order_score:.3f}"
            )
        else:
            print(f"[跳过] {idx:03d}: left_found={found_l}, right_found={found_r}")

        if save_debug:
            left_dbg = draw_corners_debug(left, board_size, found_l, corners_l)
            right_dbg = draw_corners_debug(right, board_size, found_r, corners_r)
            wide = np.hstack([left_dbg, right_dbg])
            cv2.imwrite(os.path.join(output_dir, f"corners_{idx:06d}.jpg"), wide)

    if image_size is None:
        raise RuntimeError("没有可用图片。")

    if len(object_points) < 8:
        print("[提醒] 有效双目标定图片少于 8 组，结果可能不稳定。建议至少 15~30 组。")

    if len(object_points) == 0:
        raise RuntimeError("没有任何一组图片同时检测到左右棋盘格。")

    print(f"\n有效双目标定图片数量: {len(object_points)} / {len(left_files)}")
    print(f"image_size: {image_size}")

    return object_points, left_points, right_points, image_size


def compute_rectified_output_size(image_size: Tuple[int, int], out_scale: float, out_width: int, out_height: int) -> Tuple[int, int]:
    """
    计算极线校正输出画布大小。

    对大夹角双摄：
        如果还用原始 1920x1080 输出，OpenCV 为了把旋转后的画面塞进同样大小的画布，
        往往会裁剪比较多，或者出现很强的缩放感。

    所以这里允许 out_scale > 1：
        例如 1920x1080 + out_scale=1.15 -> 2208x1242

    后续实时显示可以缩放，最终融合时再根据有效区域裁剪。
    """
    w, h = image_size

    if out_width > 0 and out_height > 0:
        return int(out_width), int(out_height)

    return max(1, int(round(w * out_scale))), max(1, int(round(h * out_scale)))


def sample_image_border_points(image_size: Tuple[int, int], samples: int = 120) -> np.ndarray:
    """
    采样原图边界点。

    只看四个角点不够，因为畸变/矫正后边缘可能是弯的。
    所以沿着四条边采样一圈点，用于估计 remap 后图像大概落在哪个区域。
    """
    w, h = image_size
    xs = np.linspace(0, w - 1, samples, dtype=np.float32)
    ys = np.linspace(0, h - 1, samples, dtype=np.float32)

    pts = []

    for x in xs:
        pts.append([x, 0])
        pts.append([x, h - 1])

    for y in ys:
        pts.append([0, y])
        pts.append([w - 1, y])

    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)


def projected_bbox_after_rectify(
    image_size: Tuple[int, int],
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    估计一张图经过 rectification 后落在输出画布中的包围盒。

    返回：
        xmin, ymin, xmax, ymax
    """
    pts = sample_image_border_points(image_size)
    proj = cv2.undistortPoints(pts, K, D, R=R, P=P).reshape(-1, 2)

    finite = np.isfinite(proj[:, 0]) & np.isfinite(proj[:, 1])
    proj = proj[finite]

    if proj.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    xmin = float(np.min(proj[:, 0]))
    ymin = float(np.min(proj[:, 1]))
    xmax = float(np.max(proj[:, 0]))
    ymax = float(np.max(proj[:, 1]))

    return xmin, ymin, xmax, ymax


def auto_center_rectified_projection(
    image_size: Tuple[int, int],
    rectified_size: Tuple[int, int],
    K_left: np.ndarray,
    D_left: np.ndarray,
    K_right: np.ndarray,
    D_right: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    自动平移 P1/P2，让 rectified 后的左右图尽量落在输出画布中心。

    为什么需要这个：
        stereoRectify + newImageSize 放大画布后，OpenCV 不一定把有效内容放在画布中心。
        结果就是图像内容被甩到边缘，棋盘格/场景被裁掉。
    
    处理策略：
        1. 估计左右图 rectified 后各自的边界 bbox。
        2. 左右相机的 x 方向可以分别平移，减少左右画面被裁。
        3. y 方向使用同一个平移量，保持极线仍然水平对齐。
    """
    rect_w, rect_h = rectified_size

    bbox_l = projected_bbox_after_rectify(image_size, K_left, D_left, R1, P1)
    bbox_r = projected_bbox_after_rectify(image_size, K_right, D_right, R2, P2)

    lx1, ly1, lx2, ly2 = bbox_l
    rx1, ry1, rx2, ry2 = bbox_r

    # x 方向左右各自居中
    shift_x_l = rect_w * 0.5 - (lx1 + lx2) * 0.5
    shift_x_r = rect_w * 0.5 - (rx1 + rx2) * 0.5

    # y 方向左右使用共同平移，保持极线对齐
    union_y1 = min(ly1, ry1)
    union_y2 = max(ly2, ry2)
    shift_y = rect_h * 0.5 - (union_y1 + union_y2) * 0.5

    P1_new = P1.copy()
    P2_new = P2.copy()

    P1_new[0, 2] += shift_x_l
    P2_new[0, 2] += shift_x_r
    P1_new[1, 2] += shift_y
    P2_new[1, 2] += shift_y

    print("  auto_center_rectified:")
    print(f"    left bbox before  : ({lx1:.1f}, {ly1:.1f}) -> ({lx2:.1f}, {ly2:.1f})")
    print(f"    right bbox before : ({rx1:.1f}, {ry1:.1f}) -> ({rx2:.1f}, {ry2:.1f})")
    print(f"    shift_x_left      : {shift_x_l:.1f}")
    print(f"    shift_x_right     : {shift_x_r:.1f}")
    print(f"    shift_y_common    : {shift_y:.1f}")

    bbox_l_after = projected_bbox_after_rectify(image_size, K_left, D_left, R1, P1_new)
    bbox_r_after = projected_bbox_after_rectify(image_size, K_right, D_right, R2, P2_new)

    print(f"    left bbox after   : ({bbox_l_after[0]:.1f}, {bbox_l_after[1]:.1f}) -> ({bbox_l_after[2]:.1f}, {bbox_l_after[3]:.1f})")
    print(f"    right bbox after  : ({bbox_r_after[0]:.1f}, {bbox_r_after[1]:.1f}) -> ({bbox_r_after[2]:.1f}, {bbox_r_after[3]:.1f})")

    return P1_new, P2_new


def run_build_mode(args: argparse.Namespace) -> None:
    """
    根据 capture_dir 中的双目标定图片生成：
        1. stereoCalibrate 得到 R / T
        2. stereoRectify 得到 R1/R2/P1/P2/Q
        3. 最终 raw -> rectified 的 remap 查找表
        4. 保存 map_file
    """
    runtime_image_size = (args.width, args.height)

    # 加载单目标定。
    # 注意：build 模式使用 K_undist + 0 畸变进行 stereoCalibrate，
    # 因为 capture 模式保存的是已单目矫正的图片。
    left_calib = load_mono_calib_and_create_maps(
        args.left_calib_file,
        runtime_image_size,
        args.mono_alpha,
        args.mono_dist_scale,
    )
    right_calib = load_mono_calib_and_create_maps(
        args.right_calib_file,
        runtime_image_size,
        args.mono_alpha,
        args.mono_dist_scale,
    )

    left_files, right_files = list_pair_images(args.capture_dir)

    object_points, left_points, right_points, stereo_image_size = collect_stereo_corners(
        left_files,
        right_files,
        args.board_cols,
        args.board_rows,
        args.square_size,
        args.output_dir,
        args.save_corner_debug,
        args.right_corner_transform,
    )

    if stereo_image_size != runtime_image_size:
        print("[警告] 采集的双目标定图片尺寸与 --width/--height 不一致。")
        print(f"  stereo images : {stereo_image_size}")
        print(f"  runtime args  : {runtime_image_size}")
        print("  将以采集图片尺寸为准，同时重新生成该尺寸对应的单目 map。")
        runtime_image_size = stereo_image_size
        left_calib = load_mono_calib_and_create_maps(
            args.left_calib_file,
            runtime_image_size,
            args.mono_alpha,
            args.mono_dist_scale,
        )
        right_calib = load_mono_calib_and_create_maps(
            args.right_calib_file,
            runtime_image_size,
            args.mono_alpha,
            args.mono_dist_scale,
        )

    # --------------------------------------------------------
    # stereoCalibrate
    # --------------------------------------------------------
    # 输入图是 mono_dist_scale=0.8 后的“部分单目矫正图”，
    # 不要再强行认为它是严格零畸变图。
    # 使用 K_undist 作为初值，让 OpenCV 微调等效内参和剩余畸变。
    D_init_left = np.zeros((5, 1), dtype=np.float64)
    D_init_right = np.zeros((5, 1), dtype=np.float64)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-6,
    )

    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    print("\n开始 stereoCalibrate...")
    print("  calib mode : CALIB_USE_INTRINSIC_GUESS")
    print("  注意：允许微调 K 和剩余畸变，不再使用 CALIB_FIX_INTRINSIC")

    rms, K_left_used, D_left_used, K_right_used, D_right_used, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_calib.K_undist.copy(),
        D_init_left.copy(),
        right_calib.K_undist.copy(),
        D_init_right.copy(),
        runtime_image_size,
        criteria=criteria,
        flags=flags,
    )

    baseline = float(np.linalg.norm(T))
    print("stereoCalibrate 完成")
    print(f"  rms      : {rms:.6f} px")
    print(f"  baseline : {baseline:.3f} mm")
    print("  K_left_used:")
    print(K_left_used)
    print("  D_left_used:")
    print(D_left_used.ravel())
    print("  K_right_used:")
    print(K_right_used)
    print("  D_right_used:")
    print(D_right_used.ravel())
    if rms > 2.0:
        print("  [提醒] rms > 2 px，极线校正可能不理想；建议继续检查图片质量、同步和坏图。")

    # --------------------------------------------------------
    # stereoRectify
    # --------------------------------------------------------
    rectified_size = compute_rectified_output_size(
        runtime_image_size,
        args.out_scale,
        args.out_width,
        args.out_height,
    )

    # 对大夹角双摄：
    #   zero_disparity=False 通常不会强制两个主点完全一样，可减少不必要的平移/裁切。
    #   如果你后续要做标准双目深度，可以打开 zero_disparity。
    rectify_flags = 0 if args.no_zero_disparity else cv2.CALIB_ZERO_DISPARITY

    print("\n开始 stereoRectify...")
    print(f"  rectify_alpha  : {args.rectify_alpha}")
    print(f"  zero_disparity : {not args.no_zero_disparity}")
    print(f"  output_size    : {rectified_size}")

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left_used,
        D_left_used,
        K_right_used,
        D_right_used,
        runtime_image_size,
        R,
        T,
        flags=rectify_flags,
        alpha=args.rectify_alpha,
        newImageSize=rectified_size,
    )

    print("stereoRectify 完成")
    print(f"  roi_left  : {tuple(int(x) for x in roi_left)}")
    print(f"  roi_right : {tuple(int(x) for x in roi_right)}")
    if args.auto_center_rectified:
        print("\n开始自动居中 rectified 画面...")
        P1, P2 = auto_center_rectified_projection(
            runtime_image_size,
            rectified_size,
            K_left_used,
            D_left_used,
            K_right_used,
            D_right_used,
            R1,
            R2,
            P1,
            P2,
        )


    # --------------------------------------------------------
    # 生成最终 raw -> rectified 的 remap
    # --------------------------------------------------------
    # 关键优化点：
    #   不保存 "undistorted -> rectified" 的 map，
    #   而是用原始 K_raw/D_raw + R1/P1 直接生成 "raw -> rectified" map。
    #   实时阶段只做一次 cv2.remap。
    print("\n生成最终 raw -> rectified remap 表...")

    left_rect_map1, left_rect_map2 = cv2.initUndistortRectifyMap(
        left_calib.K_raw,
        left_calib.D_raw,
        R1,
        P1,
        rectified_size,
        cv2.CV_16SC2,
    )

    right_rect_map1, right_rect_map2 = cv2.initUndistortRectifyMap(
        right_calib.K_raw,
        right_calib.D_raw,
        R2,
        P2,
        rectified_size,
        cv2.CV_16SC2,
    )

    # 额外保存一套 "已单目矫正图 -> 极线校正图" 的 map。
    # 这套 map 主要用于 build 阶段调试，因为 capture 模式保存的是已单目矫正图片。
    # 实时阶段仍然建议使用上面的 raw -> rectified map，只做一次 remap。
    left_undist_rect_map1, left_undist_rect_map2 = cv2.initUndistortRectifyMap(
        K_left_used,
        D_left_used,
        R1,
        P1,
        rectified_size,
        cv2.CV_16SC2,
    )

    right_undist_rect_map1, right_undist_rect_map2 = cv2.initUndistortRectifyMap(
        K_right_used,
        D_right_used,
        R2,
        P2,
        rectified_size,
        cv2.CV_16SC2,
    )

    ensure_dir(os.path.dirname(args.map_file) or ".")
    np.savez_compressed(
        args.map_file,
        # 基本尺寸
        raw_image_size=np.array(runtime_image_size, dtype=np.int32),
        rectified_size=np.array(rectified_size, dtype=np.int32),

        # 最终实时使用的 map：raw -> rectified
        left_rect_map1=left_rect_map1,
        left_rect_map2=left_rect_map2,
        right_rect_map1=right_rect_map1,
        right_rect_map2=right_rect_map2,

        # 调试用：已单目矫正图 -> 极线校正图
        left_undist_rect_map1=left_undist_rect_map1,
        left_undist_rect_map2=left_undist_rect_map2,
        right_undist_rect_map1=right_undist_rect_map1,
        right_undist_rect_map2=right_undist_rect_map2,

        # 也保存单目参数，便于复查
        left_K_raw=left_calib.K_raw,
        left_D_raw=left_calib.D_raw,
        right_K_raw=right_calib.K_raw,
        right_D_raw=right_calib.D_raw,
        left_K_undist=left_calib.K_undist,
        right_K_undist=right_calib.K_undist,

        # stereoCalibrate 结果
        K_left_used=K_left_used,
        D_left_used=D_left_used,
        K_right_used=K_right_used,
        D_right_used=D_right_used,
        R=R,
        T=T,
        E=E,
        F=F,
        rms=np.array(rms, dtype=np.float64),
        baseline=np.array(baseline, dtype=np.float64),

        # stereoRectify 结果
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi_left=np.array(roi_left, dtype=np.int32),
        roi_right=np.array(roi_right, dtype=np.int32),

        # 参数记录
        board_cols=np.array(args.board_cols, dtype=np.int32),
        board_rows=np.array(args.board_rows, dtype=np.int32),
        square_size=np.array(args.square_size, dtype=np.float64),
        mono_alpha=np.array(args.mono_alpha, dtype=np.float64),
        mono_dist_scale=np.array(args.mono_dist_scale, dtype=np.float64),
        rectify_alpha=np.array(args.rectify_alpha, dtype=np.float64),
        zero_disparity=np.array(0 if args.no_zero_disparity else 1, dtype=np.int32),
        out_scale=np.array(args.out_scale, dtype=np.float64),
    )

    print("\n已保存 remap 查找表：")
    print(f"  {args.map_file}")
    print("保存的是最终 raw -> rectified map，实时阶段只需一次 cv2.remap。")

    # --------------------------------------------------------
    # 保存一组测试图，方便立即检查极线
    # --------------------------------------------------------
    if args.save_build_preview:
        run_test_image_with_first_pair(
            args.map_file,
            left_files[0],
            right_files[0],
            args.output_dir,
            args.epiline_step,
            input_already_undistorted=True,
        )


# ============================================================
# 7. 测试 remap：图片输入
# ============================================================

def load_map_file(map_file: str):
    if not os.path.exists(map_file):
        raise RuntimeError(f"找不到 remap 文件: {map_file}")

    data = np.load(map_file)
    required = ["left_rect_map1", "left_rect_map2", "right_rect_map1", "right_rect_map2"]
    for key in required:
        if key not in data.files:
            raise RuntimeError(f"remap 文件缺少字段: {key}")

    raw_image_size = parse_image_size(data["raw_image_size"])
    rectified_size = parse_image_size(data["rectified_size"]) if "rectified_size" in data.files else (
        data["left_rect_map1"].shape[1],
        data["left_rect_map1"].shape[0],
    )

    return data, raw_image_size, rectified_size


def rectify_pair_by_map(
    left_img: np.ndarray,
    right_img: np.ndarray,
    map_data,
    input_already_undistorted: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据输入类型选择 map：

    input_already_undistorted=False:
        输入是摄像头原始图，使用 raw -> rectified map。

    input_already_undistorted=True:
        输入是 capture 模式保存的已单目畸变矫正图，使用 undistorted -> rectified map。
        这主要用于离线调试，不建议实时阶段使用。
    """
    if input_already_undistorted:
        keys = (
            "left_undist_rect_map1",
            "left_undist_rect_map2",
            "right_undist_rect_map1",
            "right_undist_rect_map2",
        )
    else:
        keys = (
            "left_rect_map1",
            "left_rect_map2",
            "right_rect_map1",
            "right_rect_map2",
        )

    for key in keys:
        if key not in map_data.files:
            raise RuntimeError(f"remap 文件缺少字段: {key}")

    left_rect = cv2.remap(
        left_img,
        map_data[keys[0]],
        map_data[keys[1]],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        right_img,
        map_data[keys[2]],
        map_data[keys[3]],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_rect, right_rect


def make_wide_debug(left_rect: np.ndarray, right_rect: np.ndarray, epiline_step: int) -> Tuple[np.ndarray, np.ndarray]:
    h = min(left_rect.shape[0], right_rect.shape[0])
    wide = np.hstack([left_rect[:h], right_rect[:h]])
    lines = wide.copy()

    for y in range(0, lines.shape[0], max(1, epiline_step)):
        cv2.line(lines, (0, y), (lines.shape[1], y), (0, 255, 255), 1)

    cv2.line(lines, (left_rect.shape[1], 0), (left_rect.shape[1], lines.shape[0]), (255, 255, 255), 2)
    return wide, lines


def run_test_image_with_first_pair(
    map_file: str,
    left_image: str,
    right_image: str,
    output_dir: str,
    epiline_step: int,
    input_already_undistorted: bool,
) -> None:
    args_like = argparse.Namespace(
        map_file=map_file,
        left_image=left_image,
        right_image=right_image,
        output_dir=output_dir,
        epiline_step=epiline_step,
        display_scale=0.3,
        headless=True,
        input_already_undistorted=input_already_undistorted,
    )
    run_test_image_mode(args_like)


def run_test_image_mode(args: argparse.Namespace) -> None:
    """
    用一对原始图片测试 map 文件。

    注意：
        map 文件是 raw -> rectified。
        所以 test-image 输入最好是原始摄像头图。
        如果你输入的是 capture 模式保存的已单目畸变矫正图，画面会被重复矫正，不建议。
    """
    map_data, raw_image_size, rectified_size = load_map_file(args.map_file)

    left = cv2.imread(args.left_image)
    right = cv2.imread(args.right_image)
    if left is None:
        raise RuntimeError(f"左图读取失败: {args.left_image}")
    if right is None:
        raise RuntimeError(f"右图读取失败: {args.right_image}")

    if (left.shape[1], left.shape[0]) != raw_image_size:
        print("[警告] 左图尺寸与 map raw_image_size 不一致，自动 resize。")
        left = cv2.resize(left, raw_image_size)
    if (right.shape[1], right.shape[0]) != raw_image_size:
        print("[警告] 右图尺寸与 map raw_image_size 不一致，自动 resize。")
        right = cv2.resize(right, raw_image_size)

    left_rect, right_rect = rectify_pair_by_map(
        left,
        right,
        map_data,
        input_already_undistorted=args.input_already_undistorted,
    )
    wide, wide_lines = make_wide_debug(left_rect, right_rect, args.epiline_step)

    ensure_dir(args.output_dir)
    cv2.imwrite(os.path.join(args.output_dir, "test_left_rect.jpg"), left_rect)
    cv2.imwrite(os.path.join(args.output_dir, "test_right_rect.jpg"), right_rect)
    cv2.imwrite(os.path.join(args.output_dir, "test_wide.jpg"), wide)
    cv2.imwrite(os.path.join(args.output_dir, "test_wide_lines.jpg"), wide_lines)

    print("测试图片已保存：")
    print(f"  {os.path.join(args.output_dir, 'test_left_rect.jpg')}")
    print(f"  {os.path.join(args.output_dir, 'test_right_rect.jpg')}")
    print(f"  {os.path.join(args.output_dir, 'test_wide.jpg')}")
    print(f"  {os.path.join(args.output_dir, 'test_wide_lines.jpg')}")

    if not args.headless:
        cv2.imshow("test wide lines", resize_for_display(wide_lines, args.display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# 8. 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline capture stereo calibration pairs and build raw-to-rectified remap maps."
    )

    parser.add_argument(
        "--mode",
        choices=["capture", "build", "test-image"],
        required=True,
        help="capture=采集已单目矫正双目标定图；build=生成 remap；test-image=用图片测试 remap",
    )

    # 摄像头参数
    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="采集/标定宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="采集/标定高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="禁用 MJPG")

    # 单目标定参数
    parser.add_argument("--left-calib-file", default=DEFAULT_LEFT_CALIB_FILE, help="左相机单目标定 npz")
    parser.add_argument("--right-calib-file", default=DEFAULT_RIGHT_CALIB_FILE, help="右相机单目标定 npz")
    parser.add_argument("--mono-alpha", type=float, default=0.0, help="单目 getOptimalNewCameraMatrix alpha")
    parser.add_argument("--mono-dist-scale", type=float, default=0.8, help="单目畸变矫正强度，建议与你之前单目脚本一致")

    # 棋盘格参数
    parser.add_argument("--board-cols", type=int, default=11, help="棋盘格内角点列数")
    parser.add_argument("--board-rows", type=int, default=8, help="棋盘格内角点行数")
    parser.add_argument("--square-size", type=float, default=22.0, help="棋盘格方格边长，单位通常是 mm")

    # 输入输出路径
    parser.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR, help="双目标定图保存/读取目录")
    parser.add_argument("--map-file", default=DEFAULT_MAP_FILE, help="输出 remap npz 文件")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="调试输出目录")
    parser.add_argument("--left-image", default="", help="test-image 模式左图")
    parser.add_argument("--right-image", default="", help="test-image 模式右图")
    parser.add_argument(
        "--input-already-undistorted",
        action="store_true",
        help="test-image 输入是否已经是 capture 模式保存的单目畸变矫正图",
    )

    # stereoRectify 优化参数
    parser.add_argument(
        "--rectify-alpha",
        type=float,
        default=1.0,
        help="stereoRectify alpha：0裁黑边/可能放大，1保留视野/黑边更多。大夹角建议先用 1.0",
    )
    parser.add_argument(
        "--zero-disparity",
        action="store_true",
        help="强制使用 CALIB_ZERO_DISPARITY。大夹角拼接一般不建议；做标准双目深度时可以打开。",
    )
    parser.add_argument("--out-scale", type=float, default=1.15, help="极线校正输出画布缩放。大夹角建议 1.1~1.4")
    parser.add_argument("--out-width", type=int, default=0, help="手动指定极线校正输出宽度，>0 时优先")
    parser.add_argument("--out-height", type=int, default=0, help="手动指定极线校正输出高度，>0 时优先")
    parser.add_argument(
        "--auto-center-rectified",
        action="store_true",
        help="stereoRectify 后自动平移 P1/P2，让左右有效画面尽量居中，减少裁切",
    )

    # 调试和显示
    parser.add_argument("--display-scale", type=float, default=0.3, help="显示缩放")
    parser.add_argument("--epiline-step", type=int, default=80, help="水平极线间隔")
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--display-every", type=int, default=1, help="每隔多少帧刷新一次显示窗口，增大可减少显示开销")
    parser.add_argument("--preview-undistorted", action="store_true", help="capture 预览单目畸变矫正后的画面；会降低 FPS")
    parser.add_argument("--capture-detect-on-save", action="store_true", help="capture 按 s 保存时检测一次棋盘格；默认不检测")
    parser.add_argument("--reject-without-board", action="store_true", help="配合 --capture-detect-on-save 使用；检测失败则不保存")
    parser.add_argument("--save-ext", default="png", choices=["png", "jpg", "jpeg"], help="capture 保存图片格式，默认 png")
    parser.add_argument("--save-raw", action="store_true", help="capture 保存已单目矫正图的同时，额外保存 raw 原图备份")
    parser.add_argument("--save-corner-debug", action="store_true", help="build 时保存角点检测调试图")
    parser.add_argument("--save-build-preview", action="store_true", help="build 后使用第一组图保存预览")
    parser.add_argument(
        "--right-corner-transform",
        default="auto_direction",
        choices=["auto_direction", "orig", "flip_ud", "flip_lr", "rot180"],
        help="右图棋盘格角点顺序修正方式。右图 rectified 倒置时，必须尝试 orig/flip_ud/flip_lr/rot180。",
    )

    args = parser.parse_args()

    # 内部统一使用 no_zero_disparity 变量：
    # 默认不使用 CALIB_ZERO_DISPARITY，更适合大夹角保留视野；
    # 用户传 --zero-disparity 时再强制使用。
    args.no_zero_disparity = not args.zero_disparity

    return args


# ============================================================
# 9. 主函数
# ============================================================

def main() -> None:
    args = parse_args()

    print("\n================ 离线双目 remap 生成脚本 ================")
    print(f"mode              : {args.mode}")
    print(f"left_calib_file   : {args.left_calib_file}")
    print(f"right_calib_file  : {args.right_calib_file}")
    print(f"capture_dir       : {args.capture_dir}")
    print(f"map_file          : {args.map_file}")
    print(f"image_size        : {(args.width, args.height)}")
    print(f"board             : {args.board_cols} x {args.board_rows}")
    print(f"square_size       : {args.square_size}")
    print(f"mono_alpha        : {args.mono_alpha}")
    print(f"mono_dist_scale   : {args.mono_dist_scale}")
    print(f"rectify_alpha     : {args.rectify_alpha}")
    print(f"zero_disparity    : {not args.no_zero_disparity}")
    print(f"out_scale         : {args.out_scale}")
    print(f"right_corner_mode : {args.right_corner_transform}")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n[提示] 未检测到 DISPLAY，OpenCV 窗口可能无法显示。")
        print("可以先执行：export DISPLAY=:0")

    if args.mode == "capture":
        run_capture_mode(args)
    elif args.mode == "build":
        run_build_mode(args)
    elif args.mode == "test-image":
        if not args.left_image or not args.right_image:
            raise RuntimeError("test-image 模式必须指定 --left-image 和 --right-image")
        run_test_image_mode(args)


if __name__ == "__main__":
    main()
