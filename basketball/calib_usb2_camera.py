#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USB2 摄像头棋盘格标定 + 实时畸变矫正程序

功能概述：
    本程序使用 OpenCV 对第二个 USB 摄像头进行棋盘格标定，获取相机内参矩阵和畸变系数，
    然后利用这些参数对摄像头画面进行实时畸变矫正。

适用场景：
    1. Linux / RK3588 / USB 摄像头
    2. OpenCV 支持 V4L2，但不支持 GStreamer
    3. 新增的第二个 USB 摄像头，设备节点可能是 /dev/video42、/dev/video43 等
    4. 棋盘格标定板：
       - 图案阵列 12 x 9 个方格
       - OpenCV 实际使用 11 x 8 个内角点

使用流程（三步走）：
    1. 采集棋盘格图片：
       python3 calib_usb2_camera.py --mode capture

    2. 根据图片计算相机内参和畸变参数：
       python3 calib_usb2_camera.py --mode calibrate

    3. 实时畸变矫正：
       python3 calib_usb2_camera.py --mode undistort

说明：
    本脚本是针对“第二个 USB 摄像头”的版本。
    不知道新摄像头挂载在哪个 /dev/videoX 上，所以把设备节点集中写在前面的宏定义区域。
    后面只需要修改 USB2_DEVICE 这一行即可。
"""

import os
import glob
import time
import argparse
import sys
import select
import termios
import tty

import cv2
import numpy as np


# =============================================================================
# USB2 摄像头宏定义区域
# =============================================================================
# 后续只需要修改这里即可，例如：
#   USB2_DEVICE = "/dev/video42"
#   USB2_DEVICE = "/dev/video43"
#   USB2_DEVICE = "/dev/video44"
#
# 可以用下面命令查看当前摄像头节点：
#   ls /dev/video*
#   v4l2-ctl --list-devices
USB2_DEVICE = "/dev/video41"

# USB2 摄像头标定图片保存目录
USB2_CALIB_IMAGE_DIR = "/home/elf/work/basketball/calib_usb2_images"

# USB2 摄像头标定参数文件
USB2_CALIB_FILE = "/home/elf/work/basketball/camera_usb2_calib.npz"

# USB2 摄像头实时畸变矫正后图片保存目录
USB2_UNDISTORT_SAVE_DIR = "/home/elf/work/basketball/stereo_undistorted_calib/Left"


# =============================================================================
# 标定板参数
# =============================================================================
# 棋盘格图案是 12 x 9 个方格（黑白相间的格子）。
# OpenCV 的 findChessboardCorners() 需要的是“内角点”数量，而不是方格数量。
# 内角点 = 方格数 - 1，所以 12 x 9 方格对应 11 x 8 内角点。
DEFAULT_BOARD_COLS = 11
DEFAULT_BOARD_ROWS = 8

# 每个方格的实际物理边长，单位 mm。
# 这个值用于将像素坐标转换为物理坐标，对畸变矫正本身无影响，
# 但会影响求出的平移向量的单位。
DEFAULT_SQUARE_SIZE = 22.0


# =============================================================================
# 摄像头默认参数
# =============================================================================
DEFAULT_DEVICE = USB2_DEVICE
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


# =============================================================================
# 亚像素角点优化参数
# =============================================================================
# 亚像素优化 cornerSubPix 可以在检测到的角点基础上，
# 进一步提高精度到亚像素级别。
SUBPIX_WIN_SIZE_FULL = (11, 11)
SUBPIX_WIN_SIZE_PREVIEW = (7, 7)

SUBPIX_CRITERIA_FULL = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

SUBPIX_CRITERIA_PREVIEW = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    20,
    0.01
)

# 棋盘格检测标志位
# CALIB_CB_ADAPTIVE_THRESH：自适应阈值，增强对光照不均匀场景的适应性。
# CALIB_CB_NORMALIZE_IMAGE：归一化图像亮度，增强对光照变化的鲁棒性。
CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE
)


class FPSCounter:
    """
    实时 FPS 计算器。

    工作原理：
        每次调用 update() 时累加帧计数，当距离上次统计时间 >= 1 秒时，
        计算 FPS = 帧数 / 经过时间，然后重置计数器。
    """

    def __init__(self):
        self._last_time = time.time()
        self._frame_count = 0
        self._fps = 0.0

    def update(self):
        """
        更新帧计数并返回当前 FPS。
        """
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_time

        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = now

        return self._fps


class TerminalKeyReader:
    """
    终端按键读取器（非阻塞模式）。

    背景：
        在 Linux 环境下，OpenCV 的图像窗口 imshow 可能无法正确获取键盘焦点，
        导致 cv2.waitKey() 读不到按键。
        此时用户按下的字符会直接进入终端。

    解决方案：
        将终端设置为 cbreak 模式，使得：
        1. 按键立即返回，不需要按 Enter
        2. 按键不会回显到终端
        3. 可以用 select() 做非阻塞读取
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
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                self.old_settings
            )

    def read_key(self):
        """
        非阻塞读取一个终端按键。
        没有按键时返回 None。
        """
        if not self.enabled:
            return None

        readable, _, _ = select.select([sys.stdin], [], [], 0)

        if readable:
            return sys.stdin.read(1)

        return None


def open_usb_camera(device, width, height, fps, use_mjpg=True):
    """
    打开 USB 摄像头。

    背景：
        本程序使用 V4L2 后端打开 USB 摄像头。
        这是因为你当前 RK3588 板端的 Python OpenCV 不支持 GStreamer，
        所以不能使用 GStreamer pipeline 打开摄像头。

    关于 MJPG 格式：
        USB 2.0 的带宽有限。
        如果使用 YUYV 未压缩格式传输 1920x1080@30fps，
        带宽需求会非常大，容易导致丢帧、卡顿或无法打开高分辨率。

        MJPG 是 JPEG 压缩格式，可以明显降低 USB 带宽压力，
        更适合 USB 摄像头在 RK3588 上使用 1080P 图像。

    参数：
        device：摄像头设备节点，例如 /dev/video42
        width：期望图像宽度
        height：期望图像高度
        fps：期望帧率
        use_mjpg：是否请求 MJPG 格式

    返回：
        cv2.VideoCapture 对象
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    # 设置缓冲区大小为 1，只保留最新的一帧，降低延迟。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头：{device}")

    if use_mjpg:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    real_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    fourcc_str = "".join([
        chr((real_fourcc >> 8 * i) & 0xFF)
        for i in range(4)
    ])

    print("摄像头已打开：")
    print(f"  device      : {device}")
    print(f"  width       : {real_width}")
    print(f"  height      : {real_height}")
    print(f"  fps         : {real_fps}")
    print(f"  fourcc      : {fourcc_str}")

    return cap


def create_object_points(board_cols, board_rows, square_size):
    """
    创建棋盘格角点在真实世界中的三维坐标。

    相机标定需要两组对应的点：
        1. object_points：棋盘格角点在真实世界中的 3D 坐标
        2. image_points：棋盘格角点在图像中的 2D 像素坐标

    对于平面棋盘格，我们假设它放在 Z=0 的平面上，
    所以所有角点的 Z 坐标都是 0。
    """
    objp = np.zeros((board_rows * board_cols, 3), np.float32)

    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)

    objp[:, :2] = grid * square_size

    return objp


def find_corners(frame, board_cols, board_rows, detect_scale=1.0):
    """
    在图像中检测棋盘格角点，并进行亚像素优化。

    检测流程：
        1. 将图像转为灰度图
        2. 可选按 detect_scale 缩放图像，加快检测速度
        3. 使用 findChessboardCornersSB 或 findChessboardCorners 粗检测角点
        4. 使用 cornerSubPix 对角点进行亚像素精化

    detect_scale：
        1.0 表示原图检测，精度最高但速度较慢。
        0.5 表示缩小一半检测，速度更快，适合实时预览。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detect_scale <= 0 or detect_scale > 1.0:
        detect_scale = 1.0

    if detect_scale != 1.0:
        detect_image = cv2.resize(
            frame,
            None,
            fx=detect_scale,
            fy=detect_scale,
            interpolation=cv2.INTER_AREA
        )
        detect_gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    else:
        detect_gray = gray

    detect_gray_eq = cv2.equalizeHist(detect_gray)

    patterns = [
        (board_cols, board_rows),
        (board_rows, board_cols),
    ]

    found = False
    corners = None

    for pattern_size in patterns:
        if hasattr(cv2, "findChessboardCornersSB"):
            flags_sb = (
                cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_EXHAUSTIVE |
                cv2.CALIB_CB_ACCURACY
            )

            found, corners = cv2.findChessboardCornersSB(
                detect_gray_eq,
                pattern_size,
                flags_sb
            )

            if found:
                break

        found, corners = cv2.findChessboardCorners(
            detect_gray_eq,
            pattern_size,
            CHESSBOARD_FLAGS
        )

        if found:
            break

    if not found or corners is None:
        return False, None, gray

    corners = cv2.cornerSubPix(
        detect_gray,
        corners,
        winSize=SUBPIX_WIN_SIZE_PREVIEW if detect_scale != 1.0 else SUBPIX_WIN_SIZE_FULL,
        zeroZone=(-1, -1),
        criteria=SUBPIX_CRITERIA_PREVIEW if detect_scale != 1.0 else SUBPIX_CRITERIA_FULL
    )

    if detect_scale != 1.0:
        corners = corners / detect_scale

    return True, corners, gray


def capture_images(args):
    """
    手动采集 USB2 摄像头标定图片。

    按键说明：
        c：保存当前检测成功的棋盘格图片
        s：直接保存当前原图，不管是否检测到角点
        q / Esc：退出采集

    图片默认保存目录：
        /home/elf/work/basketball/calib_usb2_images
    """
    os.makedirs(args.output_dir, exist_ok=True)

    cap = open_usb_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_mjpg=not args.no_mjpg
    )

    saved_count = len(glob.glob(os.path.join(args.output_dir, "calib_*.jpg")))

    print("\n采集说明：")
    print("  按 c：保存当前检测成功的棋盘格图片")
    print("  按 s：直接保存当前原图，不管是否检测到角点")
    print("  按 q 或 Esc：退出")
    if args.headless:
        print("  [headless 模式] 无显示窗口，纯终端控制")
    print("  如果按键进入终端也没关系，本程序可以读取终端按键")
    print("  建议采集 15~30 张有效图片\n")

    print("当前标定板参数：")
    print("  图案阵列：12 x 9 方格")
    print(f"  内角点：{args.board_cols} x {args.board_rows}")
    print(f"  方格边长：{args.square_size} mm\n")

    print("实时检测参数：")
    print(f"  detect_every : {args.detect_every}")
    print(f"  detect_scale : {args.detect_scale}")
    print("  如果帧率低，可以增大 --detect-every 或减小 --detect-scale")
    print("  如果一直识别不到角点，可以把 --detect-scale 改成 1.0\n")

    fps_counter = FPSCounter()

    frame_index = 0
    last_frame = None
    last_found = False
    last_corners = None

    with TerminalKeyReader() as key_reader:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("警告：读取摄像头帧失败")
                continue

            last_frame = frame.copy()
            frame_index += 1

            show_fps = fps_counter.update()

            if frame_index % args.detect_every == 0:
                last_found, last_corners, _ = find_corners(
                    frame,
                    args.board_cols,
                    args.board_rows,
                    detect_scale=args.detect_scale
                )

            if not args.headless:
                display = frame.copy()

                if last_found and last_corners is not None:
                    cv2.drawChessboardCorners(
                        display,
                        (args.board_cols, args.board_rows),
                        last_corners,
                        last_found
                    )

                    status_text = "Corners FOUND | c: save calibration image"
                    status_color = (0, 255, 0)
                else:
                    status_text = "Corners NOT found | move board closer / improve light"
                    status_color = (0, 0, 255)

                cv2.putText(
                    display,
                    status_text,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2
                )

                cv2.putText(
                    display,
                    f"Saved: {saved_count} | Preview FPS: {show_fps:.1f}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    "Keys: c=save valid | s=save raw | q/Esc=quit",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                if args.display_scale != 1.0:
                    display_show = cv2.resize(
                        display,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    display_show = display

                cv2.imshow("USB2 capture calibration images", display_show)

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

            print(f"收到按键：{repr(key_char)}")

            if key_char in ("q", "Q") or cv_key == 27:
                print("退出采集")
                break

            if key_char in ("s", "S"):
                filename = os.path.join(
                    args.output_dir,
                    f"raw_{saved_count:03d}.jpg"
                )
                cv2.imwrite(filename, last_frame)
                print(f"[RAW SAVE] 已保存原图：{filename}")
                saved_count += 1
                continue

            if key_char in ("c", "C"):
                if not last_found or last_corners is None:
                    print("[FAIL] 当前没有检测到棋盘格角点，未保存")
                    continue

                print("正在对原始分辨率图像重新检测角点...")

                found_full, corners_full, _ = find_corners(
                    last_frame,
                    args.board_cols,
                    args.board_rows,
                    detect_scale=1.0
                )

                if not found_full:
                    print("[FAIL] 预览检测到了，但全分辨率复检失败，未保存")
                    continue

                filename = os.path.join(
                    args.output_dir,
                    f"calib_{saved_count:03d}.jpg"
                )

                preview_filename = os.path.join(
                    args.output_dir,
                    f"preview_{saved_count:03d}.jpg"
                )

                cv2.imwrite(filename, last_frame)

                preview = last_frame.copy()
                cv2.drawChessboardCorners(
                    preview,
                    (args.board_cols, args.board_rows),
                    corners_full,
                    found_full
                )
                cv2.imwrite(preview_filename, preview)

                print(f"[OK] 已保存标定图：{filename}")
                print(f"[OK] 已保存预览图：{preview_filename}")

                saved_count += 1
                continue

    cap.release()
    cv2.destroyAllWindows()


def calibrate_camera(args):
    """
    根据采集到的 USB2 棋盘格图片进行相机标定。

    标定流程：
        1. 读取所有 calib_*.jpg
        2. 对每张图片检测棋盘格角点
        3. 收集 3D 世界坐标和 2D 图像坐标
        4. 调用 cv2.calibrateCamera() 求解内参矩阵和畸变系数
        5. 计算重投影误差，评估标定质量
        6. 保存标定结果到 camera_usb2_calib.npz
    """
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "calib_*.jpg")))

    if len(image_paths) == 0:
        raise RuntimeError(f"没有找到标定图片：{args.input_dir}")

    print(f"找到 {len(image_paths)} 张图片")

    objp = create_object_points(
        args.board_cols,
        args.board_rows,
        args.square_size
    )

    object_points = []
    image_points = []
    image_size = None

    for path in image_paths:
        frame = cv2.imread(path)

        if frame is None:
            print(f"[跳过] 无法读取图片：{path}")
            continue

        h, w = frame.shape[:2]
        image_size = (w, h)

        found, corners, _ = find_corners(
            frame,
            args.board_cols,
            args.board_rows,
            detect_scale=1.0
        )

        if found:
            object_points.append(objp)
            image_points.append(corners)
            print(f"[成功] {path}")
        else:
            print(f"[失败] {path}，未检测到棋盘格角点")

    valid_count = len(object_points)

    if valid_count < 8:
        raise RuntimeError(
            f"有效标定图片太少：{valid_count} 张。"
            f"建议至少 15 张，最低也应大于 8 张。"
        )

    print("\n开始 USB2 相机标定...")

    w, h = image_size

    init_camera_matrix = np.array([
        [w, 0, w / 2.0],
        [0, w, h / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float64)

    init_dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        init_camera_matrix,
        init_dist_coeffs,
        flags=flags
    )

    print("\n================ USB2 标定结果 ================")

    print("\nRMS 误差：")
    print(rms)

    print("\n相机内参矩阵 camera_matrix：")
    print(camera_matrix)

    print("\n畸变系数 dist_coeffs：")
    print(dist_coeffs)

    total_error = 0.0

    for i in range(valid_count):
        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        error = cv2.norm(
            image_points[i],
            projected_points,
            cv2.NORM_L2
        ) / len(projected_points)

        total_error += error

    mean_error = total_error / valid_count

    print("\n平均重投影误差 mean reprojection error：")
    print(mean_error)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        args.alpha,
        image_size
    )

    print("\n矫正后新内参矩阵 new_camera_matrix：")
    print(new_camera_matrix)

    print("\n有效区域 ROI：")
    print(roi)

    np.savez(
        args.output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        new_camera_matrix=new_camera_matrix,
        roi=np.array(roi),
        image_size=np.array(image_size),
        board_cols=np.array(args.board_cols),
        board_rows=np.array(args.board_rows),
        square_size=np.array(args.square_size),
        rms=np.array(rms),
        mean_error=np.array(mean_error)
    )

    print(f"\nUSB2 标定参数已保存到：{args.output}")

    print("\n误差参考：")
    print("  < 0.3 像素：很好")
    print("  0.3 ~ 0.8 像素：正常可用")
    print("  0.8 ~ 1.5 像素：勉强可用")
    print("  > 1.5 像素：建议重新采集标定图片")


def undistort_live(args):
    """
    USB2 摄像头实时畸变矫正。

    功能：
        读取 USB2 的标定参数文件：
            /home/elf/work/basketball/camera_usb2_calib.npz

        对 USB2 摄像头画面进行实时畸变矫正。

    按键说明：
        q / Esc：退出
        s：保存当前矫正图像
        o：切换上下对比显示 / 只显示矫正图

    矫正后图片默认保存到：
        /home/elf/work/basketball/stereo_undistorted_calib/Left_usb2
    """
    if not os.path.exists(args.calib_file):
        raise RuntimeError(f"找不到标定文件：{args.calib_file}")

    data = np.load(args.calib_file)

    camera_matrix = data["camera_matrix"]
    dist_coeffs_raw = data["dist_coeffs"]
    saved_image_size = tuple(data["image_size"].astype(int))

    dist_shape = dist_coeffs_raw.shape
    dist_flat = dist_coeffs_raw.reshape(-1).copy()

    # 只缩放径向畸变系数 k1、k2、k3。
    # 不缩放 p1、p2，因为它们是切向畸变系数。
    if len(dist_flat) >= 1:
        dist_flat[0] *= args.dist_scale

    if len(dist_flat) >= 2:
        dist_flat[1] *= args.dist_scale

    if len(dist_flat) >= 5:
        dist_flat[4] *= args.dist_scale

    dist_coeffs = dist_flat.reshape(dist_shape)

    print(f"\nDistortion scale: {args.dist_scale}")
    print("Original dist coeffs:")
    print(dist_coeffs_raw)
    print("Scaled dist coeffs:")
    print(dist_coeffs)

    print("读取 USB2 标定参数：")
    print(f"  calib_file       : {args.calib_file}")
    print(f"  saved image size : {saved_image_size}")

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nDist coeffs:")
    print(dist_coeffs)

    cap = open_usb_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_mjpg=not args.no_mjpg
    )

    current_image_size = (args.width, args.height)

    if current_image_size != saved_image_size:
        print("\n警告：当前分辨率和标定分辨率不一致！")
        print(f"  标定分辨率：{saved_image_size}")
        print(f"  当前分辨率：{current_image_size}")
        print("  建议用相同分辨率进行标定和畸变矫正。\n")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        current_image_size,
        args.alpha,
        current_image_size
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        current_image_size,
        cv2.CV_16SC2
    )

    print("\n开始 USB2 实时畸变矫正")
    print("  q / Esc：退出")
    print("  s：保存当前矫正图像")
    print("  o：切换上下对比显示 / 只显示矫正图")
    print(f"  保存目录：{args.save_dir}")
    print("  如果按键显示在终端里也没关系，本函数支持终端按键\n")

    fps_counter = FPSCounter()
    save_count = len(glob.glob(os.path.join(args.save_dir, "undistorted_usb2_*.jpg")))
    show_original = args.show_original

    os.makedirs(args.save_dir, exist_ok=True)

    with TerminalKeyReader() as key_reader:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("警告：读取摄像头帧失败")
                continue

            undistorted = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR
            )

            if args.crop:
                x, y, w, h = roi
                undistorted = undistorted[y:y + h, x:x + w]

            show_fps = fps_counter.update()

            if not args.headless:
                if show_original:
                    original = frame.copy()

                    if original.shape[:2] != undistorted.shape[:2]:
                        original = cv2.resize(
                            original,
                            (undistorted.shape[1], undistorted.shape[0])
                        )

                    original_show = original.copy()
                    undistorted_show = undistorted.copy()

                    cv2.putText(
                        original_show,
                        f"USB2 Original | FPS: {show_fps:.1f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        undistorted_show,
                        "USB2 Undistorted | q/Esc quit | s save | o toggle",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

                    separator = np.full(
                        (8, original_show.shape[1], 3),
                        255,
                        dtype=np.uint8
                    )

                    combined = np.vstack(
                        (original_show, separator, undistorted_show)
                    )

                    if args.display_scale != 1.0:
                        combined = cv2.resize(
                            combined,
                            None,
                            fx=args.display_scale,
                            fy=args.display_scale,
                            interpolation=cv2.INTER_AREA
                        )

                    cv2.imshow("USB2 original(top) | undistorted(bottom)", combined)

                else:
                    display = undistorted.copy()

                    cv2.putText(
                        display,
                        f"USB2 Undistorted | FPS: {show_fps:.1f} | q/Esc quit | s save | o toggle",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    if args.display_scale != 1.0:
                        display = cv2.resize(
                            display,
                            None,
                            fx=args.display_scale,
                            fy=args.display_scale,
                            interpolation=cv2.INTER_AREA
                        )

                    cv2.imshow("USB2 undistorted", display)

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

            print(f"收到按键：{repr(key_char)}")

            if key_char in ("q", "Q") or cv_key == 27:
                print("退出 USB2 实时矫正")
                break

            if key_char in ("s", "S"):
                filename = os.path.join(
                    args.save_dir,
                    f"undistorted_usb2_{save_count:03d}.jpg"
                )
                cv2.imwrite(filename, undistorted)
                print(f"[SAVE] {filename}")
                save_count += 1
                continue

            if key_char in ("o", "O"):
                show_original = not show_original
                print(f"show_original = {show_original}")
                continue

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    """
    解析命令行参数。

    主要参数：
        --mode：
            capture   采集 USB2 标定图片
            calibrate 计算 USB2 标定参数
            undistort USB2 实时畸变矫正

        --device：
            USB2 摄像头设备节点。
            默认使用文件开头 USB2_DEVICE 宏定义。

        --output-dir：
            标定图片保存目录。
            默认 /home/elf/work/basketball/calib_usb2_images

        --input-dir：
            标定图片输入目录。
            默认 /home/elf/work/basketball/calib_usb2_images

        --output：
            标定结果输出文件。
            默认 /home/elf/work/basketball/camera_usb2_calib.npz

        --calib-file：
            实时矫正使用的标定文件。
            默认 /home/elf/work/basketball/camera_usb2_calib.npz

        --save-dir：
            实时矫正后按 s 保存图片的目录。
            默认 /home/elf/work/basketball/stereo_undistorted_calib/Left_usb2
    """
    parser = argparse.ArgumentParser(
        description="USB2 camera chessboard calibration and undistortion"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["capture", "calibrate", "undistort"],
        help="capture=采集标定图片, calibrate=计算标定参数, undistort=实时畸变矫正"
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="USB2 摄像头设备节点，例如 /dev/video42"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="摄像头采集宽度"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="摄像头采集高度"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="摄像头帧率"
    )

    parser.add_argument(
        "--no-mjpg",
        action="store_true",
        help="不请求 MJPG，改用摄像头默认格式。一般不建议开启。"
    )

    parser.add_argument(
        "--board-cols",
        type=int,
        default=DEFAULT_BOARD_COLS,
        help="棋盘格横向内角点数量。12x9 方格应填 11。"
    )

    parser.add_argument(
        "--board-rows",
        type=int,
        default=DEFAULT_BOARD_ROWS,
        help="棋盘格纵向内角点数量。12x9 方格应填 8。"
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE,
        help="方格边长，单位 mm。"
    )

    parser.add_argument(
        "--output-dir",
        default=USB2_CALIB_IMAGE_DIR,
        help="USB2 采集标定图片保存目录"
    )

    parser.add_argument(
        "--input-dir",
        default=USB2_CALIB_IMAGE_DIR,
        help="USB2 标定图片输入目录"
    )

    parser.add_argument(
        "--output",
        default=USB2_CALIB_FILE,
        help="USB2 标定结果输出文件"
    )

    parser.add_argument(
        "--calib-file",
        default=USB2_CALIB_FILE,
        help="USB2 实时矫正使用的标定文件"
    )

    parser.add_argument(
        "--save-dir",
        default=USB2_UNDISTORT_SAVE_DIR,
        help="USB2 矫正后图片保存目录"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="矫正视野参数。0=裁黑边，1=保留全部视野"
    )

    parser.add_argument(
        "--dist-scale",
        type=float,
        default=0.8,
        help="畸变矫正强度。1.0=原始标定结果，0.8=减弱矫正，0=不矫正。"
    )

    parser.add_argument(
        "--crop",
        action="store_true",
        help="矫正后是否裁掉黑边"
    )

    parser.add_argument(
        "--show-original",
        action="store_true",
        help="实时矫正时是否显示原图和矫正图对比"
    )

    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.5,
        help="显示缩放比例。1920x1080 屏幕太大时建议 0.5"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式，不显示 OpenCV 窗口"
    )

    parser.add_argument(
        "--detect-every",
        type=int,
        default=5,
        help="每隔多少帧检测一次棋盘格角点"
    )

    parser.add_argument(
        "--detect-scale",
        type=float,
        default=0.5,
        help="实时预览检测时的缩放比例。0.5 表示缩小一半检测，1.0 表示原图检测。"
    )

    return parser.parse_args()


def main():
    """
    程序入口。

    根据 --mode 参数分发：
        capture   → 采集 USB2 标定图片
        calibrate → 计算 USB2 标定参数
        undistort → USB2 实时畸变矫正
    """
    args = parse_args()

    print("当前 USB2 参数：")
    print(f"  mode        : {args.mode}")
    print(f"  device      : {args.device}")
    print(f"  size        : {args.width}x{args.height}")
    print(f"  fps         : {args.fps}")
    print(f"  board       : {args.board_cols} x {args.board_rows} inner corners")
    print(f"  square size : {args.square_size} mm")
    print(f"  calib images: {args.output_dir}")
    print(f"  calib file  : {args.calib_file}")
    print(f"  save dir    : {args.save_dir}")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n[提示] 未检测到 DISPLAY 环境变量，OpenCV 窗口可能无法显示。")
        print("  请先运行：export DISPLAY=:0")
        print("  然后重新执行本程序。\n")

    if args.mode == "capture":
        capture_images(args)

    elif args.mode == "calibrate":
        calibrate_camera(args)

    elif args.mode == "undistort":
        undistort_live(args)


if __name__ == "__main__":
    main()