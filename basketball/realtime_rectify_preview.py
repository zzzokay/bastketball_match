#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
realtime_rectify_preview.py

用途：
    实时阶段使用。
    读取离线生成的 remap 查找表，对左右摄像头原始 RGB 图做双目极线校正，然后输出左右并排预览图。

当前脚本故意不做这些事情：
    1. 不转灰度估计 overlap_px
    2. 不估计 vertical_offset
    3. 不生成 alpha mask
    4. 不做亮度补偿
    5. 不做最终融合

原因：
    你当前要求是先把“单目畸变矫正 + 双目极线校正 + remap 查找表实时使用”这一步做好。
    后续确认极线校正效果、黑边、视野保留情况后，再加 overlap / 裁剪 / 融合会更稳。

输入：
    stereo_rectify_maps_wide.npz
    该文件建议由 offline_build_stereo_rectify_maps.py 生成。
    它包含：
        left_rect_map1 / left_rect_map2
        right_rect_map1 / right_rect_map2
        raw_image_size
        rectified_size

输出：
    左右双目极线校正后的并排宽图：
        [left_rect | right_rect]

典型用法：
    python3 realtime_rectify_preview.py \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --width 1920 \
        --height 1080 \
        --fps 30 \
        --display-scale 0.25 \
        --output-dir /home/elf/work/basketball/stereo_rectify_live_debug

按键：
    q / Esc : 退出
    s       : 保存当前 raw / rectified / wide / wide_lines
    l       : 开关水平极线显示
"""

import argparse
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"
DEFAULT_MAP_FILE = "/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz"
DEFAULT_OUTPUT_DIR = "/home/elf/work/basketball/stereo_rectify_live_debug"

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


# ============================================================
# 1. 小工具
# ============================================================

class TerminalKeyReader:
    """
    非阻塞终端按键读取器。

    有些 RK3588 桌面环境下，OpenCV 窗口不一定能拿到键盘焦点。
    这个类可以直接从终端读取 q/s/l。
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


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 格式不正确: {value}")
    return int(flat[0]), int(flat[1])


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def read_key_from_cv_and_terminal(cv_key: int, terminal_key: Optional[str]) -> Optional[str]:
    if terminal_key is not None:
        return terminal_key
    if cv_key != -1:
        return chr(cv_key & 0xFF)
    return None


def open_usb_camera(device: str, width: int, height: int, fps: int, use_mjpg: bool = True) -> cv2.VideoCapture:
    """
    使用 V4L2 打开 USB 摄像头。

    注意：
        双路 1080P@30fps 用 MJPG 通常更稳定。
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
        print("[警告] 实际采集尺寸与请求尺寸不同。后续会 resize 到 remap 的 raw_image_size，但最好让采集分辨率一致。")

    return cap


# ============================================================
# 2. remap 查找表数据结构
# ============================================================

@dataclass
class RectifyMapPack:
    """
    双目极线校正 remap 查找表。

    raw_image_size:
        摄像头原始输入尺寸，格式 (width, height)。

    rectified_size:
        remap 输出尺寸，格式 (width, height)。
        如果离线时 out_scale > 1，这个尺寸会大于原始尺寸。

    left_map1/left_map2/right_map1/right_map2:
        原始图 raw -> 双目极线校正图 rectified 的映射表。
    """

    raw_image_size: Tuple[int, int]
    rectified_size: Tuple[int, int]
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray
    roi_left: Optional[Tuple[int, int, int, int]]
    roi_right: Optional[Tuple[int, int, int, int]]
    rms: Optional[float]
    baseline: Optional[float]
    rectify_alpha: Optional[float]
    zero_disparity: Optional[bool]


def load_rectify_maps(map_file: str) -> RectifyMapPack:
    """
    加载离线生成的 remap 文件。

    兼容两种字段命名：
        新脚本：
            left_rect_map1 / left_rect_map2
            right_rect_map1 / right_rect_map2

        你之前上传的旧脚本可能保存：
            left_map1 / left_map2
            right_map1 / right_map2
    """
    if not os.path.exists(map_file):
        raise RuntimeError(f"找不到 remap 文件: {map_file}")

    data = np.load(map_file)

    if "left_rect_map1" in data.files:
        left_map1 = data["left_rect_map1"]
        left_map2 = data["left_rect_map2"]
        right_map1 = data["right_rect_map1"]
        right_map2 = data["right_rect_map2"]
    elif "left_map1" in data.files:
        print("[兼容模式] 检测到旧字段 left_map1/right_map1。")
        print("  注意：旧 map 可能是 undistorted -> rectified，不一定适合直接输入 raw 摄像头图。")
        left_map1 = data["left_map1"]
        left_map2 = data["left_map2"]
        right_map1 = data["right_map1"]
        right_map2 = data["right_map2"]
    else:
        raise RuntimeError("remap 文件中没有找到 left_rect_map1 或 left_map1。")

    if "raw_image_size" in data.files:
        raw_image_size = parse_image_size(data["raw_image_size"])
    elif "image_size" in data.files:
        raw_image_size = parse_image_size(data["image_size"])
    else:
        # 没有显式保存时，按 map 输出尺寸兜底。
        raw_image_size = (left_map1.shape[1], left_map1.shape[0])

    if "rectified_size" in data.files:
        rectified_size = parse_image_size(data["rectified_size"])
    else:
        rectified_size = (left_map1.shape[1], left_map1.shape[0])

    roi_left = tuple(int(x) for x in data["roi_left"].reshape(-1)) if "roi_left" in data.files else None
    roi_right = tuple(int(x) for x in data["roi_right"].reshape(-1)) if "roi_right" in data.files else None

    rms = float(data["rms"]) if "rms" in data.files else None
    baseline = float(data["baseline"]) if "baseline" in data.files else None
    rectify_alpha = float(data["rectify_alpha"]) if "rectify_alpha" in data.files else None
    zero_disparity = bool(int(data["zero_disparity"])) if "zero_disparity" in data.files else None

    print("已加载 remap 文件:")
    print(f"  {map_file}")
    print(f"  raw_image_size  : {raw_image_size}")
    print(f"  rectified_size  : {rectified_size}")
    print(f"  left_map shape  : {left_map1.shape}")
    print(f"  right_map shape : {right_map1.shape}")
    if roi_left is not None:
        print(f"  roi_left        : {roi_left}")
    if roi_right is not None:
        print(f"  roi_right       : {roi_right}")
    if rms is not None:
        print(f"  stereo rms      : {rms:.6f} px")
    if baseline is not None:
        print(f"  baseline        : {baseline:.3f} mm")
    if rectify_alpha is not None:
        print(f"  rectify_alpha   : {rectify_alpha}")
    if zero_disparity is not None:
        print(f"  zero_disparity  : {zero_disparity}")

    return RectifyMapPack(
        raw_image_size=raw_image_size,
        rectified_size=rectified_size,
        left_map1=left_map1,
        left_map2=left_map2,
        right_map1=right_map1,
        right_map2=right_map2,
        roi_left=roi_left,
        roi_right=roi_right,
        rms=rms,
        baseline=baseline,
        rectify_alpha=rectify_alpha,
        zero_disparity=zero_disparity,
    )


# ============================================================
# 3. 图像处理
# ============================================================

def rectify_raw_pair(raw_left: np.ndarray, raw_right: np.ndarray, maps: RectifyMapPack) -> Tuple[np.ndarray, np.ndarray]:
    """
    对原始左右图做双目极线校正。

    关键点：
        离线生成的是 raw -> rectified map，
        所以实时阶段只需要一次 cv2.remap。
    """
    left_rect = cv2.remap(
        raw_left,
        maps.left_map1,
        maps.left_map2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        raw_right,
        maps.right_map1,
        maps.right_map2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_rect, right_rect


def make_side_by_side(left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
    """
    只做并排预览，不做融合。

    当前阶段输出：
        [left_rect | right_rect]

    后续你确认极线校正效果后，再在这个基础上加入：
        overlap_px
        vertical_offset
        crop
        brightness gain
        alpha blend
    """
    h = min(left_rect.shape[0], right_rect.shape[0])
    return np.hstack([left_rect[:h], right_rect[:h]])


def draw_epipolar_lines(wide_img: np.ndarray, left_width: int, step: int) -> np.ndarray:
    """
    给并排图画水平线，方便检查极线校正效果。

    判断方法：
        同一个棋盘格角点 / 球场线特征 / 远处物体，
        在左右图里应该尽量落在同一条水平线上。
    """
    debug = wide_img.copy()
    h, w = debug.shape[:2]

    for y in range(0, h, max(1, step)):
        cv2.line(debug, (0, y), (w, y), (0, 255, 255), 1)

    cv2.line(debug, (left_width, 0), (left_width, h), (255, 255, 255), 2)
    return debug


def maybe_resize_to_raw_size(img: np.ndarray, raw_image_size: Tuple[int, int], name: str) -> np.ndarray:
    """
    如果摄像头实际输出尺寸和 map 文件的 raw_image_size 不一致，则 resize。

    注意：
        resize 只是兜底。
        最好让摄像头实际输出尺寸、单目标定尺寸、双目标定尺寸完全一致。
    """
    if (img.shape[1], img.shape[0]) == raw_image_size:
        return img

    print(f"[警告] {name} 输入尺寸 {img.shape[1]}x{img.shape[0]} 与 map raw_image_size {raw_image_size} 不一致，自动 resize。")
    return cv2.resize(img, raw_image_size, interpolation=cv2.INTER_AREA)


def save_debug_frames(
    output_dir: str,
    prefix: str,
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    wide: np.ndarray,
    wide_lines: np.ndarray,
    save_raw: bool,
) -> None:
    ensure_dir(output_dir)

    if save_raw:
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_raw_left.jpg"), raw_left)
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_raw_right.jpg"), raw_right)

    cv2.imwrite(os.path.join(output_dir, f"{prefix}_left_rect.jpg"), left_rect)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_right_rect.jpg"), right_rect)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_wide.jpg"), wide)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_wide_lines.jpg"), wide_lines)

    print("已保存当前帧：")
    print(f"  {os.path.join(output_dir, f'{prefix}_left_rect.jpg')}")
    print(f"  {os.path.join(output_dir, f'{prefix}_right_rect.jpg')}")
    print(f"  {os.path.join(output_dir, f'{prefix}_wide.jpg')}")
    print(f"  {os.path.join(output_dir, f'{prefix}_wide_lines.jpg')}")


# ============================================================
# 4. 实时主流程
# ============================================================

def run_live(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    maps = load_rectify_maps(args.map_file)

    if (args.width, args.height) != maps.raw_image_size:
        print("\n[提醒] 当前请求采集尺寸和 map 文件 raw_image_size 不一致。")
        print(f"  capture request : {(args.width, args.height)}")
        print(f"  map raw size    : {maps.raw_image_size}")
        print("  建议把 --width/--height 设置为 map raw_image_size，避免 resize 影响精度。\n")

    cap_left = open_usb_camera(args.left_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)
    cap_right = open_usb_camera(args.right_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)

    fps_counter = FPSCounter()
    save_idx = 0
    show_lines = not args.no_lines

    print("\n开始实时 remap 预览")
    print("当前阶段只做：raw RGB -> remap 极线校正 -> 左右并排显示")
    print("不做 overlap / crop / 亮度补偿 / alpha 融合")
    print("按键：q/Esc 退出，s 保存当前帧，l 开关水平线。")

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

            raw_left = maybe_resize_to_raw_size(raw_left, maps.raw_image_size, "left")
            raw_right = maybe_resize_to_raw_size(raw_right, maps.raw_image_size, "right")

            left_rect, right_rect = rectify_raw_pair(raw_left, raw_right, maps)

            wide = make_side_by_side(left_rect, right_rect)
            wide_lines = draw_epipolar_lines(wide, left_rect.shape[1], args.epiline_step)

            fps = fps_counter.update()

            display = wide_lines if show_lines else wide
            display = display.copy()

            cv2.putText(
                display,
                f"Rectify preview | FPS:{fps:.1f} | q quit | s save | l lines:{show_lines}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            if not args.headless:
                cv2.imshow("realtime stereo rectify preview", resize_for_display(display, args.display_scale))
                cv_key = cv2.waitKey(1)
            else:
                cv_key = -1

            key = read_key_from_cv_and_terminal(cv_key, key_reader.read_key())

            if key is None:
                continue

            if key in ("q", "Q") or cv_key == 27:
                print("退出实时预览。")
                break

            if key in ("l", "L"):
                show_lines = not show_lines
                print(f"水平线显示: {show_lines}")

            if key in ("s", "S"):
                prefix = f"live_{save_idx:06d}"
                save_debug_frames(
                    args.output_dir,
                    prefix,
                    raw_left,
                    raw_right,
                    left_rect,
                    right_rect,
                    wide,
                    wide_lines,
                    save_raw=args.save_raw,
                )
                save_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


# ============================================================
# 5. 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime preview using precomputed stereo rectification remap maps.")

    parser.add_argument("--map-file", default=DEFAULT_MAP_FILE, help="离线生成的 remap npz 文件")
    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")

    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="摄像头采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="禁用 MJPG")

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="保存调试图片目录")
    parser.add_argument("--display-scale", type=float, default=0.3, help="显示缩放比例")
    parser.add_argument("--epiline-step", type=int, default=80, help="水平极线间隔")
    parser.add_argument("--no-lines", action="store_true", help="默认不显示水平极线")
    parser.add_argument("--save-raw", action="store_true", help="按 s 保存时同时保存 raw 原图")
    parser.add_argument("--headless", action="store_true", help="无窗口运行，只处理按键和保存")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ 实时 remap 预览脚本 ================")
    print(f"map_file      : {args.map_file}")
    print(f"left_device   : {args.left_device}")
    print(f"right_device  : {args.right_device}")
    print(f"capture size  : {(args.width, args.height)}")
    print(f"fps           : {args.fps}")
    print(f"output_dir    : {args.output_dir}")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n[提示] 未检测到 DISPLAY，OpenCV 窗口可能无法显示。")
        print("可以先执行：export DISPLAY=:0")

    run_live(args)


if __name__ == "__main__":
    main()
