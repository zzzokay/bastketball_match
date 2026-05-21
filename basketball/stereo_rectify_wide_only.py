#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stereo_rectify_wide_only.py

只做一件事：
    读取左右图像 -> 单目畸变矫正（可选） -> 双目极线校正 -> 左右并排成一张宽图

它不做：
    1. SIFT / ORB 特征匹配
    2. Homography 估计
    3. warpPerspective 全景融合
    4. 接缝融合

适合你当前阶段：
    先验证 stereo_calibrate_from_undistorted.npz 是否能把左右图极线校正好。

两种使用场景：

一、处理已经畸变矫正后的左右图片：
    python3 stereo_rectify_wide_only.py \
        --mode image \
        --left-image  /home/elf/work/basketball/stereo_undistorted_calib/Left/align_000.jpg \
        --right-image /home/elf/work/basketball/stereo_undistorted_calib/Right/align_000.jpg \
        --stereo-file /home/elf/work/basketball/stereo_undistorted_calib/stereo_calibrate_from_undistorted.npz \
        --images-already-undistorted \
        --output-dir /home/elf/work/basketball/stereo_wide_debug

二、实时读取两个原始 USB 摄像头，先单目畸变矫正，再双目极线校正：
    python3 stereo_rectify_wide_only.py \
        --mode live \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
        --right-calib-file /home/elf/work/basketball/camera_calib.npz \
        --stereo-file /home/elf/work/basketball/stereo_undistorted_calib/stereo_calibrate_from_undistorted.npz \
        --output-dir /home/elf/work/basketball/stereo_wide_debug \
        --display-scale 0.25

按键：
    q / Esc : 退出
    s       : 保存当前 left_rect / right_rect / wide / wide_with_lines
"""

import argparse
import os
import sys
import time
import select
import termios
import tty

import cv2
import numpy as np


DEFAULT_STEREO_FILE = "/home/elf/work/basketball/stereo_undistorted_calib/stereo_calibrate_from_undistorted.npz"
DEFAULT_LEFT_CALIB_FILE = "/home/elf/work/basketball/camera_usb2_calib.npz"
DEFAULT_RIGHT_CALIB_FILE = "/home/elf/work/basketball/camera_calib.npz"
DEFAULT_OUTPUT_DIR = "/home/elf/work/basketball/stereo_wide_debug"

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


class TerminalKeyReader:
    """在 OpenCV 窗口拿不到键盘焦点时，也可以从终端读 q/s 等按键。"""

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

    def read_key(self):
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1)
        return None


class FPSCounter:
    """实时 FPS 统计。"""

    def __init__(self):
        self.last_time = time.time()
        self.count = 0
        self.fps = 0.0

    def update(self):
        self.count += 1
        now = time.time()
        dt = now - self.last_time
        if dt >= 1.0:
            self.fps = self.count / dt
            self.count = 0
            self.last_time = now
        return self.fps


def open_usb_camera(device, width, height, fps, use_mjpg=True):
    """使用 V4L2 打开 USB 摄像头。"""
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
    print(f"  size  : {real_w}x{real_h}")
    print(f"  fps   : {real_fps}")
    print(f"  fourcc: {fourcc_str}")

    return cap


def scale_dist_coeffs(dist_coeffs_raw, dist_scale):
    """
    按你之前单目畸变矫正脚本的逻辑缩放径向畸变系数。

    OpenCV 常见畸变系数顺序：
        [k1, k2, p1, p2, k3]

    这里只缩放 k1/k2/k3，不缩放 p1/p2。
    这样可以和你原来的 calib_usb_camera.py / calib_usb2_camera.py 保持一致。
    """
    D = dist_coeffs_raw.copy()
    flat = D.reshape(-1)

    if len(flat) >= 1:
        flat[0] *= dist_scale
    if len(flat) >= 2:
        flat[1] *= dist_scale
    if len(flat) >= 5:
        flat[4] *= dist_scale

    return flat.reshape(dist_coeffs_raw.shape)


def create_mono_undistort_maps(calib_file, image_size, alpha=0.0, dist_scale=0.8):
    """
    加载单目标定文件，生成单目畸变矫正 map。

    输入：
        原始 USB 摄像头图像

    输出：
        已畸变矫正图像

    注意：
        你的 stereo_calibrate_from_undistorted.npz 是用“已畸变矫正后的图片”算出来的，
        所以实时 raw 摄像头输入必须先经过这个单目矫正步骤。
    """
    if not os.path.exists(calib_file):
        raise RuntimeError(f"找不到单目标定文件: {calib_file}")

    data = np.load(calib_file)
    K = data["camera_matrix"]
    D_raw = data["dist_coeffs"]
    saved_size = tuple(data["image_size"].astype(int))

    if saved_size != image_size:
        print("[警告] 单目标定分辨率和当前运行分辨率不一致。")
        print(f"  calib : {saved_size}")
        print(f"  input : {image_size}")
        print("  建议采集、单目标定、双目标定、运行都使用同一分辨率。")

    D = scale_dist_coeffs(D_raw, dist_scale)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, image_size, alpha, image_size)

    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        D,
        None,
        new_K,
        image_size,
        cv2.CV_16SC2,
    )

    print(f"单目畸变矫正 map 已创建: {calib_file}")
    print(f"  alpha      : {alpha}")
    print(f"  dist_scale : {dist_scale}")
    print(f"  roi        : {roi}")

    return map1, map2


def create_stereo_rectify_maps(stereo_file, rectify_alpha=0.0, zero_disparity=True):
    """
    加载 stereo_calibrate_from_undistorted.npz，生成双目极线校正 map。

    stereo npz 里应该包含：
        image_size
        K_left, D_left
        K_right, D_right
        R, T

    stereoRectify 的作用：
        把左右图旋转到同一个虚拟成像平面上，让对应点尽量落在同一水平线。
    """
    if not os.path.exists(stereo_file):
        raise RuntimeError(f"找不到双目标定文件: {stereo_file}")

    data = np.load(stereo_file)

    image_size = tuple(data["image_size"].astype(int))
    K_left = data["K_left"]
    D_left = data["D_left"]
    K_right = data["K_right"]
    D_right = data["D_right"]
    R = data["R"]
    T = data["T"]

    rms_stereo = float(data["rms_stereo"]) if "rms_stereo" in data.files else None
    baseline = float(np.linalg.norm(T))

    flags = cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        R,
        T,
        flags=flags,
        alpha=rectify_alpha,
        newImageSize=image_size,
    )

    # stereo_calibrate_from_undistorted.npz 是基于已畸变矫正图计算的，
    # 所以这里的 D_left/D_right 通常接近 0。
    # initUndistortRectifyMap 在这里主要完成“极线校正旋转”。
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        K_left,
        D_left,
        R1,
        P1,
        image_size,
        cv2.CV_16SC2,
    )

    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        K_right,
        D_right,
        R2,
        P2,
        image_size,
        cv2.CV_16SC2,
    )

    print("双目极线校正 map 已创建")
    print(f"  stereo_file : {stereo_file}")
    print(f"  image_size  : {image_size}")
    print(f"  baseline    : {baseline:.3f} mm")
    if rms_stereo is not None:
        print(f"  rms_stereo  : {rms_stereo:.4f} px")
        if rms_stereo > 2.0:
            print("  [提醒] rms_stereo > 2 px，极线校正效果可能不理想，建议后续重新检查标定质量。")
    print(f"  roi_left    : {tuple(int(x) for x in roi_left)}")
    print(f"  roi_right   : {tuple(int(x) for x in roi_right)}")

    rectify_params = {
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi_left": tuple(int(x) for x in roi_left),
        "roi_right": tuple(int(x) for x in roi_right),
        "baseline": baseline,
        "rms_stereo": rms_stereo,
    }

    return image_size, left_map1, left_map2, right_map1, right_map2, rectify_params


def rectify_pair(left_img, right_img, left_map1, left_map2, right_map1, right_map2):
    """对左右图分别执行双目极线校正。"""
    left_rect = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    return left_rect, right_rect


def make_wide_image(left_rect, right_rect):
    """
    左右并排成一张宽图。

    这不是全景融合，只是：
        [left_rect | right_rect]
    """
    h = min(left_rect.shape[0], right_rect.shape[0])
    left = left_rect[:h]
    right = right_rect[:h]
    return np.hstack([left, right])


def draw_epipolar_debug(wide_img, left_width, step=80):
    """
    在宽图上画水平线，方便检查极线是否对齐。

    看法：
        同一个物体在左图和右图中的对应位置，应该尽量落在同一条黄线附近。
    """
    debug = wide_img.copy()
    h, w = debug.shape[:2]

    for y in range(0, h, step):
        cv2.line(debug, (0, y), (w, y), (0, 255, 255), 1)

    # 中间白线表示左图和右图分界线。
    cv2.line(debug, (left_width, 0), (left_width, h), (255, 255, 255), 2)

    return debug


def save_outputs(output_dir, prefix, left_rect, right_rect, wide, wide_lines):
    """保存输出图片。"""
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_left_rect.jpg"), left_rect)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_right_rect.jpg"), right_rect)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_wide.jpg"), wide)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_wide_lines.jpg"), wide_lines)

    print("已保存：")
    print(os.path.join(output_dir, f"{prefix}_left_rect.jpg"))
    print(os.path.join(output_dir, f"{prefix}_right_rect.jpg"))
    print(os.path.join(output_dir, f"{prefix}_wide.jpg"))
    print(os.path.join(output_dir, f"{prefix}_wide_lines.jpg"))


def resize_for_display(img, scale):
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def run_image_mode(args):
    """处理一对图片，输出极线校正后的并排宽图。"""
    os.makedirs(args.output_dir, exist_ok=True)

    image_size, left_smap1, left_smap2, right_smap1, right_smap2, _ = create_stereo_rectify_maps(
        args.stereo_file,
        rectify_alpha=args.rectify_alpha,
        zero_disparity=not args.no_zero_disparity,
    )

    left = cv2.imread(args.left_image)
    right = cv2.imread(args.right_image)

    if left is None:
        raise RuntimeError(f"左图读取失败: {args.left_image}")
    if right is None:
        raise RuntimeError(f"右图读取失败: {args.right_image}")

    # 尺寸必须和 stereo npz 中的 image_size 一致。
    if (left.shape[1], left.shape[0]) != image_size:
        print("[警告] 左图尺寸和 stereo npz 不一致，自动 resize。")
        left = cv2.resize(left, image_size)
    if (right.shape[1], right.shape[0]) != image_size:
        print("[警告] 右图尺寸和 stereo npz 不一致，自动 resize。")
        right = cv2.resize(right, image_size)

    # 如果输入图已经是你之前单目畸变矫正保存出来的图，就不需要再做单目矫正。
    if not args.images_already_undistorted:
        left_mmap1, left_mmap2 = create_mono_undistort_maps(
            args.left_calib_file,
            image_size,
            alpha=args.mono_alpha,
            dist_scale=args.mono_dist_scale,
        )
        right_mmap1, right_mmap2 = create_mono_undistort_maps(
            args.right_calib_file,
            image_size,
            alpha=args.mono_alpha,
            dist_scale=args.mono_dist_scale,
        )
        left = cv2.remap(left, left_mmap1, left_mmap2, cv2.INTER_LINEAR)
        right = cv2.remap(right, right_mmap1, right_mmap2, cv2.INTER_LINEAR)

    # 双目极线校正。
    left_rect, right_rect = rectify_pair(
        left,
        right,
        left_smap1,
        left_smap2,
        right_smap1,
        right_smap2,
    )

    # 左右并排成宽图。
    wide = make_wide_image(left_rect, right_rect)
    wide_lines = draw_epipolar_debug(wide, left_rect.shape[1], step=args.epiline_step)

    save_outputs(args.output_dir, "image", left_rect, right_rect, wide, wide_lines)

    if not args.headless:
        cv2.imshow("rectified wide", resize_for_display(wide, args.display_scale))
        cv2.imshow("rectified wide with epipolar lines", resize_for_display(wide_lines, args.display_scale))
        print("按任意键关闭窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_live_mode(args):
    """实时读取两个 USB 摄像头，输出极线校正后的并排宽图。"""
    os.makedirs(args.output_dir, exist_ok=True)

    image_size, left_smap1, left_smap2, right_smap1, right_smap2, _ = create_stereo_rectify_maps(
        args.stereo_file,
        rectify_alpha=args.rectify_alpha,
        zero_disparity=not args.no_zero_disparity,
    )

    if (args.width, args.height) != image_size:
        print("[警告] 当前采集分辨率和 stereo npz 中 image_size 不一致。")
        print(f"  capture: {(args.width, args.height)}")
        print(f"  stereo : {image_size}")
        print("  建议用完全相同分辨率运行。")

    # 实时 raw 摄像头输入，默认先做单目畸变矫正。
    # 只有当前级已经输出“畸变矫正图”时，才加 --skip-mono-undistort。
    if args.skip_mono_undistort:
        left_mmap1 = left_mmap2 = None
        right_mmap1 = right_mmap2 = None
        print("已跳过单目畸变矫正：假设输入已经是已畸变矫正图。")
    else:
        left_mmap1, left_mmap2 = create_mono_undistort_maps(
            args.left_calib_file,
            image_size,
            alpha=args.mono_alpha,
            dist_scale=args.mono_dist_scale,
        )
        right_mmap1, right_mmap2 = create_mono_undistort_maps(
            args.right_calib_file,
            image_size,
            alpha=args.mono_alpha,
            dist_scale=args.mono_dist_scale,
        )

    cap_left = open_usb_camera(args.left_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)
    cap_right = open_usb_camera(args.right_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)

    fps_counter = FPSCounter()
    save_idx = 0

    print("\n开始实时极线校正宽图显示")
    print("按键：q/Esc 退出，s 保存当前帧。")

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

            # 如果实际采集尺寸不一致，先 resize 到 stereo 标定尺寸。
            # 最好不要依赖 resize，尽量让摄像头实际输出就是 1920x1080。
            if (raw_left.shape[1], raw_left.shape[0]) != image_size:
                raw_left = cv2.resize(raw_left, image_size)
            if (raw_right.shape[1], raw_right.shape[0]) != image_size:
                raw_right = cv2.resize(raw_right, image_size)

            if not args.skip_mono_undistort:
                left_undist = cv2.remap(raw_left, left_mmap1, left_mmap2, cv2.INTER_LINEAR)
                right_undist = cv2.remap(raw_right, right_mmap1, right_mmap2, cv2.INTER_LINEAR)
            else:
                left_undist = raw_left
                right_undist = raw_right

            left_rect, right_rect = rectify_pair(
                left_undist,
                right_undist,
                left_smap1,
                left_smap2,
                right_smap1,
                right_smap2,
            )

            wide = make_wide_image(left_rect, right_rect)
            wide_lines = draw_epipolar_debug(wide, left_rect.shape[1], step=args.epiline_step)

            fps = fps_counter.update()

            if not args.headless:
                display = wide_lines.copy()
                cv2.putText(
                    display,
                    f"Rectified Wide | FPS:{fps:.1f} | q quit | s save",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("rectified wide with epipolar lines", resize_for_display(display, args.display_scale))

            cv_key = -1
            if not args.headless:
                cv_key = cv2.waitKey(1)

            term_key = key_reader.read_key()
            key_char = None
            if cv_key != -1:
                key_char = chr(cv_key & 0xFF)
            if term_key is not None:
                key_char = term_key

            if key_char is None:
                continue

            if key_char in ("q", "Q") or cv_key == 27:
                print("退出实时极线校正宽图")
                break

            if key_char in ("s", "S"):
                prefix = f"live_{save_idx:06d}"
                save_outputs(args.output_dir, prefix, left_rect, right_rect, wide, wide_lines)
                save_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Only stereo rectify and make side-by-side wide image")

    parser.add_argument("--mode", choices=["image", "live"], required=True, help="image=处理一对图片；live=实时双摄像头")

    parser.add_argument("--stereo-file", default=DEFAULT_STEREO_FILE, help="stereo_calibrate_from_undistorted.npz")
    parser.add_argument("--left-calib-file", default=DEFAULT_LEFT_CALIB_FILE, help="左相机单目标定 npz")
    parser.add_argument("--right-calib-file", default=DEFAULT_RIGHT_CALIB_FILE, help="右相机单目标定 npz")

    parser.add_argument("--left-image", default="", help="image 模式左图")
    parser.add_argument("--right-image", default="", help="image 模式右图")
    parser.add_argument("--images-already-undistorted", action="store_true", help="image 输入图是否已经单目畸变矫正")

    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="采集宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="采集高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="禁用 MJPG")
    parser.add_argument("--skip-mono-undistort", action="store_true", help="live 输入已经是畸变矫正图时使用")

    parser.add_argument("--mono-alpha", type=float, default=0.0, help="单目 getOptimalNewCameraMatrix alpha")
    parser.add_argument("--mono-dist-scale", type=float, default=0.8, help="单目畸变矫正强度，建议和采集双目标定图时一致")

    parser.add_argument("--rectify-alpha", type=float, default=0.0, help="stereoRectify alpha：0裁黑边，1保留视野，-1自动")
    parser.add_argument("--no-zero-disparity", action="store_true", help="不使用 CALIB_ZERO_DISPARITY")
    parser.add_argument("--epiline-step", type=int, default=80, help="水平极线间隔")

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--display-scale", type=float, default=0.3, help="显示缩放比例")
    parser.add_argument("--headless", action="store_true", help="无显示窗口")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n================ 当前参数 ================")
    print(f"mode              : {args.mode}")
    print(f"stereo_file       : {args.stereo_file}")
    print(f"left_calib_file   : {args.left_calib_file}")
    print(f"right_calib_file  : {args.right_calib_file}")
    print(f"output_dir        : {args.output_dir}")
    print(f"mono_dist_scale   : {args.mono_dist_scale}")
    print(f"rectify_alpha     : {args.rectify_alpha}")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n[提示] 未检测到 DISPLAY，OpenCV 窗口可能无法显示。")
        print("可以先执行：export DISPLAY=:0")

    if args.mode == "image":
        if not args.left_image or not args.right_image:
            raise RuntimeError("image 模式必须指定 --left-image 和 --right-image")
        run_image_mode(args)
    elif args.mode == "live":
        run_live_mode(args)


if __name__ == "__main__":
    main()
