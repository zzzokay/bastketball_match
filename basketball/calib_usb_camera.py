#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USB 摄像头棋盘格标定 + 实时畸变矫正程序

功能概述：
    本程序使用 OpenCV 对 USB 摄像头进行棋盘格标定，获取相机内参矩阵和畸变系数，
    然后利用这些参数对摄像头画面进行实时畸变矫正。

适用场景：
    1. Linux / RK3588 / USB 摄像头
    2. OpenCV 支持 V4L2，但不支持 GStreamer
    3. 摄像头节点例如 /dev/video41
    4. 棋盘格标定板：
       - 方格边长 3 mm/22mm
       - 图案阵列 12 x 9 个方格
       - OpenCV 实际使用 11 x 8 个内角点

使用流程（三步走）：
    1. 采集棋盘格图片：
       python3 calib_usb_camera.py --mode capture

    2. 根据图片计算相机内参和畸变参数：
       python3 calib_usb_camera.py --mode calibrate

    3. 实时畸变矫正：
       python3 calib_usb_camera.py --mode undistort

相机标定原理简述：
    相机镜头（尤其是广角/鱼眼镜头）会引入径向畸变和切向畸变，导致图像中的直线变弯曲。
    通过拍摄已知几何结构的棋盘格标定板，利用 OpenCV 的张正友标定法，可以从多张图片中
    反推出相机的内参矩阵（焦距、主点）和畸变系数（k1, k2, k3 径向 + p1, p2 切向），
    然后用这些参数对图像做反向映射，消除畸变。
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
# 标定板参数
# =============================================================================
# 棋盘格图案是 12 x 9 个方格（黑白相间的格子）。
# OpenCV 的 findChessboardCorners() 需要的是"内角点"数量，而不是方格数量。
# 内角点 = 方格数 - 1，所以 12 x 9 方格对应 11 x 8 内角点。
#
# 什么是内角点？
#   想象一个 3x3 的棋盘格（4 个方格），中间有 1 个点是四个方格共享的角，
#   这就是内角点。边上的角点不算内角点。
DEFAULT_BOARD_COLS = 11          # 横向内角点数量（= 方格横向数 - 1）
DEFAULT_BOARD_ROWS = 8           # 纵向内角点数量（= 方格纵向数 - 1）

# 每个方格的实际物理边长，单位 mm。
# 这个值用于将像素坐标转换为物理坐标，对畸变矫正本身无影响，
# 但会影响求出的平移向量的单位。
# DEFAULT_SQUARE_SIZE = 3.0
DEFAULT_SQUARE_SIZE = 22.0
# USB 摄像头实时畸变矫正后图片保存目录
USB_UNDISTORT_SAVE_DIR = "/home/elf/work/basketball/stereo_undistorted_calib/Right"
# =============================================================================
# 摄像头默认参数
# =============================================================================
DEFAULT_DEVICE = "/dev/video43"  # Linux 摄像头设备节点路径
DEFAULT_WIDTH = 1920             # 采集图像宽度（像素）
DEFAULT_HEIGHT = 1080            # 采集图像高度（像素）
DEFAULT_FPS = 30                 # 采集帧率


# =============================================================================
# 亚像素角点优化参数
# =============================================================================
# 亚像素优化（cornerSubPix）可以在检测到的角点基础上，进一步提高精度到亚像素级别。
# 原理：在角点周围的小窗口内，沿梯度方向迭代搜索，使角点位置更加精确。
#
# winSize：搜索窗口大小。窗口越大，抗噪声能力越强，但精度可能降低。
# zeroZone：零区域大小，(-1, -1) 表示不使用零区域。
# criteria：迭代终止条件，包含最大迭代次数和最小误差变化量。
#
# 全分辨率标定使用更严格的参数（大窗口、多次迭代、低误差阈值），
# 实时预览使用较宽松的参数（小窗口、少迭代）以保证帧率。
SUBPIX_WIN_SIZE_FULL = (11, 11)       # 全分辨率检测的搜索窗口
SUBPIX_WIN_SIZE_PREVIEW = (7, 7)      # 实时预览的搜索窗口（更小，更快）

SUBPIX_CRITERIA_FULL = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,  # 同时满足精度或迭代次数时停止
    30,       # 最大迭代次数
    0.001     # 最小误差变化量（像素）
)

SUBPIX_CRITERIA_PREVIEW = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    20,       # 预览时迭代次数更少，提高帧率
    0.01      # 预览时误差阈值更宽松
)

# 棋盘格检测标志位
# CALIB_CB_ADAPTIVE_THRESH：使用自适应阈值二值化，增强对光照不均匀场景的适应性。
#   原理：将图像分成多个小块，每块独立计算阈值，避免全局阈值在光照不均时失效。
# CALIB_CB_NORMALIZE_IMAGE：归一化图像亮度，进一步增强对光照变化的鲁棒性。
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

    用法：
        fps_counter = FPSCounter()
        while True:
            # ... 处理一帧 ...
            fps = fps_counter.update()
            print(f"FPS: {fps:.1f}")
    """

    def __init__(self):
        """初始化 FPS 计算器。"""
        self._last_time = time.time()   # 上次统计的时间点
        self._frame_count = 0           # 当前统计周期内的帧数
        self._fps = 0.0                 # 最近一次计算得到的 FPS

    def update(self):
        """
        更新帧计数并返回当前 FPS。

        返回：
            float：当前 FPS 值。如果不到 1 秒，返回上一次的 FPS 值。
        """
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_time

        # 每隔至少 1 秒重新计算一次 FPS
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = now

        return self._fps


class TerminalKeyReader:
    """
    终端按键读取器（非阻塞模式）。

    背景：
        在 Linux 环境下，OpenCV 的图像窗口（imshow）可能无法正确获取键盘焦点，
        导致 cv2.waitKey() 读不到按键。此时用户按下的字符会直接进入终端。

    解决方案：
        将终端设置为 cbreak 模式（也叫 raw 模式的简化版），使得：
        1. 按键立即返回，不需要按 Enter
        2. 按键不会回显到终端
        3. 可以用 select() 做非阻塞读取

    注意：
        退出时必须恢复终端设置，否则终端会显示异常（如不回显输入的字符）。

    用法：
        with TerminalKeyReader() as key_reader:
            while True:
                key = key_reader.read_key()
                if key == 'q':
                    break
    """

    def __init__(self):
        self.enabled = False        # 是否成功启用了 cbreak 模式
        self.old_settings = None    # 保存原始终端设置，用于退出时恢复

    def __enter__(self):
        """
        进入上下文管理器时，将终端切换为 cbreak 模式。

        只有当 stdin 是终端（而不是管道或重定向）时才启用。
        """
        if sys.stdin.isatty():
            self.enabled = True
            # 保存当前终端设置
            self.old_settings = termios.tcgetattr(sys.stdin)
            # 设置为 cbreak 模式：字符级输入，不需要按 Enter
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文管理器时，恢复原始终端设置。

        这一步非常重要！如果不恢复，终端可能处于异常状态。
        """
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,  # 等待输出完成后再修改设置
                self.old_settings
            )

    def read_key(self):
        """
        非阻塞读取一个终端按键。

        返回：
            str 或 None：读到的字符，没有按键时返回 None。
        """
        if not self.enabled:
            return None

        # select() 检查 stdin 是否有数据可读，超时设为 0（立即返回）
        readable, _, _ = select.select([sys.stdin], [], [], 0)

        if readable:
            return sys.stdin.read(1)

        return None


def open_usb_camera(device, width, height, fps, use_mjpg=True):
    """
    打开 USB 摄像头。

    背景：
        本程序的 OpenCV 编译时没有 GStreamer 支持（编译信息中 GStreamer = NO），
        因此不能使用 GStreamer pipeline 打开摄像头，只能使用 V4L2 后端。

    关于 MJPG 格式：
        USB 2.0 的带宽有限（约 480 Mbps 理论值，实际更低）。
        如果使用 YUYV（未压缩）格式传输 1920x1080@30fps 的图像：
            1920 * 1080 * 2 (YUYV每像素2字节) * 30 = 约 124 MB/s = 992 Mbps
        这远超 USB 2.0 带宽，会导致丢帧或无法设置高分辨率。

        MJPG 是 JPEG 压缩格式，每帧压缩后通常只有 50~200 KB，
        大幅降低 USB 带宽需求，使 1920x1080@30fps 在 USB 2.0 上也能工作。

    参数：
        device (str)：摄像头设备节点路径，例如 "/dev/video41"
        width (int)：期望的图像宽度（像素）
        height (int)：期望的图像高度（像素）
        fps (int)：期望的帧率
        use_mjpg (bool)：是否请求 MJPG 格式，默认 True

    返回：
        cv2.VideoCapture：已配置好的摄像头对象

    异常：
        RuntimeError：无法打开摄像头时抛出
    """
    # 使用 V4L2 后端打开摄像头
    # cv2.CAP_V4L2 指定使用 Video4Linux2 API（Linux 标准摄像头接口）
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    # 设置缓冲区大小为 1，只保留最新的一帧。
    # 默认缓冲区可能有 3~5 帧，处理慢时会导致画面严重延迟。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头：{device}")

    # 请求摄像头使用 MJPG 编码格式
    if use_mjpg:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # 设置分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 读回实际设置值（摄像头可能无法完全满足请求，需要确认实际值）
    real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    real_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # FOURCC 是 4 字节编码，这里把它转成人类可读的字符串
    # 例如 MJPG -> "MJPG"，YUYV -> "YUYV"
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
        1. object_points：棋盘格角点在真实世界中的 3D 坐标（单位：mm）
        2. image_points：棋盘格角点在图像中的 2D 像素坐标

    对于平面棋盘格，我们假设它放在 Z=0 的平面上（即标定板平放在桌面上），
    所以所有角点的 Z 坐标都是 0。

    举例（board_cols=3, board_rows=2, square_size=3.0mm）：
        角点 0: (0.0, 0.0, 0.0)
        角点 1: (3.0, 0.0, 0.0)
        角点 2: (6.0, 0.0, 0.0)
        角点 3: (0.0, 3.0, 0.0)
        角点 4: (3.0, 3.0, 0.0)
        角点 5: (6.0, 3.0, 0.0)

    参数：
        board_cols (int)：横向内角点数量
        board_rows (int)：纵向内角点数量
        square_size (float)：方格边长（单位 mm）

    返回：
        numpy.ndarray：shape = (board_rows * board_cols, 3)，float32 类型
    """
    # 创建一个 (N, 3) 的零数组，N = 内角点总数
    objp = np.zeros((board_rows * board_cols, 3), np.float32)

    # np.mgrid 生成二维网格坐标
    # 例如 board_cols=3, board_rows=2 时：
    #   np.mgrid[0:3, 0:2] 的 shape 是 (2, 3, 2)，即：
    #   [[(0,0), (1,0), (2,0)],
    #    [(0,1), (1,1), (2,1)]]
    # .T 转置后 shape 变为 (2, 3, 2) -> (3, 2, 2)
    # .reshape(-1, 2) 展平后得到 6 个 (x, y) 点
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)

    # 将网格索引乘以方格实际边长，得到真实物理坐标
    objp[:, :2] = grid * square_size

    return objp


def find_corners(frame, board_cols, board_rows, detect_scale=1.0):
    """
    在图像中检测棋盘格角点，并进行亚像素优化。

    检测流程：
        1. 将图像转为灰度图（角点检测只需要灰度信息）
        2. （可选）按 detect_scale 缩放图像，加快检测速度
        3. 使用 findChessboardCorners() 粗检测角点
        4. 使用 cornerSubPix() 对角点进行亚像素精化

    关于 detect_scale 参数：
        当 detect_scale < 1.0 时，先将图像缩小再检测，然后把检测到的角点坐标
        放大回原图尺寸。这样可以显著提高检测速度，但精度会略有下降。
        - 实时预览建议用 0.5（缩小一半检测，速度快）
        - 标定保存时建议用 1.0（原图检测，精度高）

    参数：
        frame (numpy.ndarray)：BGR 彩色图像
        board_cols (int)：横向内角点数量
        board_rows (int)：纵向内角点数量
        detect_scale (float)：检测时的缩放比例，范围 (0, 1.0]，默认 1.0

    返回：
        tuple: (found, corners, gray)
            found (bool)：是否检测到角点
            corners (numpy.ndarray 或 None)：角点坐标，shape = (N, 1, 2)，未检测到时为 None
            gray (numpy.ndarray)：灰度图（原始尺寸）
    """
    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 如果需要缩放，使用 INTER_AREA 插值（缩小时效果最好，不会产生摩尔纹）
    if detect_scale <= 0 or detect_scale > 1.0:
        detect_scale = 1.0

    if detect_scale != 1.0:
        detect_image = cv2.resize(
            frame,
            None,                    # 不指定目标尺寸，由缩放因子决定
            fx=detect_scale,
            fy=detect_scale,
            interpolation=cv2.INTER_AREA
        )
        detect_gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    else:
        detect_gray = gray

    # 增强对比度，棋盘格检测更稳定
    detect_gray_eq = cv2.equalizeHist(detect_gray)

    patterns = [
        (board_cols, board_rows),
        (board_rows, board_cols),
    ]

    found = False
    corners = None

    for pattern_size in patterns:
        # 优先使用新版 SB 棋盘格检测，更适合高分辨率/畸变/光照变化
        if hasattr(cv2, "findChessboardCornersSB"):
            flags_sb = (
                cv2.CALIB_CB_NORMALIZE_IMAGE |
                cv2.CALIB_CB_EXHAUSTIVE |
                cv2.CALIB_CB_ACCURACY
            )
            found, corners = cv2.findChessboardCornersSB(
                detect_gray_eq, pattern_size, flags_sb
            )
            if found:
                break

        # 回退到老版检测
        found, corners = cv2.findChessboardCorners(
            detect_gray_eq, pattern_size, CHESSBOARD_FLAGS
        )
        if found:
            break

    if not found or corners is None:
        return False, None, gray

    # 亚像素角点优化
    # 在粗检测的基础上，在每个角点周围的小窗口内迭代搜索更精确的位置
    # 使用全分辨率参数（大窗口、多次迭代）以保证标定精度
    corners = cv2.cornerSubPix(
        detect_gray,
        corners,
        winSize=SUBPIX_WIN_SIZE_PREVIEW if detect_scale != 1.0 else SUBPIX_WIN_SIZE_FULL,
        zeroZone=(-1, -1),
        criteria=SUBPIX_CRITERIA_PREVIEW if detect_scale != 1.0 else SUBPIX_CRITERIA_FULL
    )

    # 如果是在缩小图上检测的，需要把角点坐标放大回原图尺寸
    if detect_scale != 1.0:
        corners = corners / detect_scale

    return True, corners, gray


def capture_images(args):
    """
    手动采集标定图片。

    功能：
        打开摄像头实时预览画面，用户可以手动控制何时保存图片。
        画面上会实时显示棋盘格角点的检测结果，方便用户判断标定板位置是否合适。

    按键说明：
        c — 保存当前检测到角点的标定图（会先对原始分辨率重新检测确认）
        s — 直接保存当前原图（不检测角点，用于快速保存）
        q / Esc — 退出采集

    性能优化：
        1. 不是每一帧都做角点检测，而是每隔 detect_every 帧检测一次
        2. 检测时可以使用缩小的图像（detect_scale），提高实时显示帧率
        3. 保存标定图时仍然保存原始 1920x1080 图像，不受预览缩放影响

    参数：
        args (argparse.Namespace)：命令行参数，包含 device, width, height, fps, output_dir 等
    """
    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)

    # 打开摄像头
    cap = open_usb_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_mjpg=not args.no_mjpg
    )

    # 统计已有标定图片数量（用于自动编号）
    saved_count = len(glob.glob(os.path.join(args.output_dir, "calib_*.jpg")))

    print("\n采集说明：")
    print("  按 c：保存当前检测成功的棋盘格图片")
    print("  按 s：直接保存当前原图，不管是否检测到角点")
    print("  按 q 或 Esc：退出")
    if args.headless:
        print("  [headless 模式] 无显示窗口，纯终端控制")
    print("  如果按键进入终端也没关系，本程序现在也能读取终端按键")
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

    # 初始化 FPS 计算器
    fps_counter = FPSCounter()

    frame_index = 0               # 帧计数器，用于控制每隔多少帧检测一次
    last_frame = None             # 最近一帧的原始图像（用于保存）
    last_found = False            # 最近一次检测是否找到角点
    last_corners = None           # 最近一次检测到的角点坐标

    # 使用 TerminalKeyReader 同时支持 OpenCV 窗口按键和终端按键
    with TerminalKeyReader() as key_reader:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("警告：读取摄像头帧失败")
                continue

            # 保存原始帧的副本（用于后续保存标定图）
            last_frame = frame.copy()
            frame_index += 1

            # 计算并显示 FPS
            show_fps = fps_counter.update()

            # 每隔 detect_every 帧检测一次角点
            # 这样可以避免每帧都做昂贵的角点检测，保证实时预览的流畅度
            if frame_index % args.detect_every == 0:
                last_found, last_corners, _ = find_corners(
                    frame,
                    args.board_cols,
                    args.board_rows,
                    detect_scale=args.detect_scale
                )

            # ==================== 画面绘制 ====================
            if not args.headless:
                display = frame.copy()

                if last_found and last_corners is not None:
                    # 在画面上绘制检测到的棋盘格角点
                    # drawChessboardCorners 会画出角点和连接线
                    cv2.drawChessboardCorners(
                        display,
                        (args.board_cols, args.board_rows),
                        last_corners,
                        last_found
                    )

                    status_text = "Corners FOUND | c: save calibration image"
                    status_color = (0, 255, 0)  # 绿色 = 检测成功
                else:
                    status_text = "Corners NOT found | move board closer / improve light"
                    status_color = (0, 0, 255)  # 红色 = 未检测到

                # 显示状态信息
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
                    (255, 255, 0),  # 青色
                    2
                )

                cv2.putText(
                    display,
                    "Keys: c=save valid | s=save raw | q/Esc=quit",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),  # 白色
                    2
                )

                # 缩放显示画面（不影响保存的原始图像）
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

                cv2.imshow("capture calibration images", display_show)

            # ==================== 按键处理 ====================
            # 同时读取 OpenCV 窗口按键和终端按键，取先收到的那个
            cv_key = -1
            if not args.headless:
                cv_key = cv2.waitKey(1)

            term_key = key_reader.read_key()

            key_char = None

            if cv_key != -1:
                # OpenCV 返回的按键码，取低 8 位转为字符
                key_char = chr(cv_key & 0xFF)

            if term_key is not None:
                key_char = term_key

            if key_char is None:
                continue

            print(f"收到按键：{repr(key_char)}")

            # q / Q / Esc：退出
            if key_char in ("q", "Q") or cv_key == 27:
                print("退出采集")
                break

            # s / S：直接保存原图（不检测角点）
            if key_char in ("s", "S"):
                filename = os.path.join(
                    args.output_dir,
                    f"raw_{saved_count:03d}.jpg"
                )
                cv2.imwrite(filename, last_frame)
                print(f"[RAW SAVE] 已保存原图：{filename}")
                saved_count += 1
                continue

            # c / C：保存检测成功的标定图
            if key_char in ("c", "C"):
                # 先检查预览时是否检测到了角点
                if not last_found or last_corners is None:
                    print("[FAIL] 当前没有检测到棋盘格角点，未保存")
                    continue

                # 为了保证标定精度，对原始全分辨率图像再做一次角点检测
                # （预览时可能用的是缩小图，精度较低）
                print("正在对原始分辨率图像重新检测角点...")

                found_full, corners_full, _ = find_corners(
                    last_frame,
                    args.board_cols,
                    args.board_rows,
                    detect_scale=1.0  # 原图检测，保证精度
                )

                if not found_full:
                    print("[FAIL] 预览检测到了，但全分辨率复检失败，未保存")
                    continue

                # 保存原始标定图
                filename = os.path.join(
                    args.output_dir,
                    f"calib_{saved_count:03d}.jpg"
                )

                # 保存角点预览图（带有角点标记，方便后续检查）
                preview_filename = os.path.join(
                    args.output_dir,
                    f"preview_{saved_count:03d}.jpg"
                )

                cv2.imwrite(filename, last_frame)

                # 在预览图上绘制角点
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

    # 释放摄像头资源，关闭所有 OpenCV 窗口
    cap.release()
    cv2.destroyAllWindows()


def calibrate_camera(args):
    """
    根据采集到的棋盘格图片进行相机标定。

    标定流程：
        1. 读取所有标定图片（calib_*.jpg）
        2. 对每张图片检测棋盘格角点
        3. 收集所有图片的 3D 世界坐标和 2D 图像坐标
        4. 调用 cv2.calibrateCamera() 求解内参矩阵和畸变系数
        5. 计算重投影误差，评估标定质量
        6. 保存标定结果到 .npz 文件

    标定结果说明：
        camera_matrix：3x3 相机内参矩阵
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
            fx, fy：焦距（像素单位）
            cx, cy：主点坐标（通常接近图像中心）

        dist_coeffs：畸变系数
            [k1, k2, p1, p2, k3]
            k1, k2, k3：径向畸变系数（桶形/枕形畸变）
            p1, p2：切向畸变系数（镜头与传感器不平行导致）

        new_camera_matrix：矫正后的新内参矩阵
            由 getOptimalNewCameraMatrix() 计算，用于 undistort 时的映射。

    参数：
        args (argparse.Namespace)：命令行参数，包含 input_dir, output, board_cols, board_rows 等
    """
    # 查找所有标定图片
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "calib_*.jpg")))

    if len(image_paths) == 0:
        raise RuntimeError(f"没有找到标定图片：{args.input_dir}")

    print(f"找到 {len(image_paths)} 张图片")

    # 生成棋盘格角点的 3D 世界坐标（所有图片共用同一组）
    objp = create_object_points(
        args.board_cols,
        args.board_rows,
        args.square_size
    )

    # 收集所有图片的 3D 世界坐标和 2D 图像坐标
    object_points = []   # 每张图对应的 3D 角点坐标
    image_points = []    # 每张图检测到的 2D 像素角点坐标
    image_size = None    # 图像尺寸 (width, height)

    for path in image_paths:
        frame = cv2.imread(path)

        if frame is None:
            print(f"[跳过] 无法读取图片：{path}")
            continue

        h, w = frame.shape[:2]
        image_size = (w, h)

        # 检测棋盘格角点（使用原图，不做缩放）
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

    # 标定至少需要多少张图片？
    # 理论上 3 张图片即可求解，但实际中需要更多来提高精度和稳定性。
    # 建议 15~30 张，最少不应低于 8 张。
    if valid_count < 8:
        raise RuntimeError(
            f"有效标定图片太少：{valid_count} 张。"
            f"建议至少 15 张，最低也应大于 8 张。"
        )

    print("\n开始相机标定...")

    # ==================== 设置初始内参猜测 ====================
    # 初始内参矩阵的设定：
    #   fx = fy = w（图像宽度）：这是一个粗略的焦距估计
    #     原理：对于视角约 60 度的镜头，焦距（像素）约等于图像宽度
    #     这只是一个初始值，calibrateCamera 会在优化过程中调整它
    #   cx = w/2, cy = h/2：主点设在图像中心（通常接近真实值）
    w, h = image_size

    init_camera_matrix = np.array([
        [w, 0, w / 2.0],
        [0, w, h / 2.0],    # 注意：这里 fy 也用 w，因为通常 fx ≈ fy
        [0, 0, 1.0]
    ], dtype=np.float64)

    # 初始畸变系数设为全零（无畸变），由优化算法自行求解
    init_dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    # 标定标志位：
    #   CALIB_USE_INTRINSIC_GUESS：使用我们提供的初始内参作为起点（加速收敛）
    #   CALIB_FIX_PRINCIPAL_POINT：固定主点不优化（避免主点和焦距耦合导致不稳定）
    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    # ==================== 执行标定 ====================
    # cv2.calibrateCamera 使用 Levenberg-Marquardt 优化算法，
    # 同时求解内参矩阵、畸变系数、每张图的旋转和平移向量。
    #
    # 返回值：
    #   rms：RMS 重投影误差（单位：像素），越小越好
    #   camera_matrix：3x3 内参矩阵
    #   dist_coeffs：畸变系数
    #   rvecs：每张图的旋转向量（Rodrigues 格式）
    #   tvecs：每张图的平移向量
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        init_camera_matrix,
        init_dist_coeffs,
        flags=flags
    )

    print("\n================ 标定结果 ================")

    print("\nRMS 误差：")
    print(rms)

    print("\n相机内参矩阵 camera_matrix：")
    print(camera_matrix)

    print("\n畸变系数 dist_coeffs：")
    print(dist_coeffs)

    # ==================== 计算平均重投影误差 ====================
    # 重投影误差：将 3D 角点通过标定得到的参数投影回图像，
    # 与实际检测到的 2D 角点比较，计算欧氏距离。
    # 这是评估标定质量的核心指标。
    total_error = 0.0

    for i in range(valid_count):
        # 将 3D 点投影到图像平面
        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        # 计算投影点与实际检测点之间的平均欧氏距离
        error = cv2.norm(
            image_points[i],
            projected_points,
            cv2.NORM_L2
        ) / len(projected_points)

        total_error += error

    mean_error = total_error / valid_count

    print("\n平均重投影误差 mean reprojection error：")
    print(mean_error)

    # ==================== 计算矫正后的新内参矩阵 ====================
    # getOptimalNewCameraMatrix 计算用于 undistort 的新内参矩阵。
    #
    # alpha 参数的含义：
    #   alpha = 0：裁掉所有黑边，有效图像区域最大，但视野变小
    #   alpha = 1：保留全部原始视野，但边缘可能有黑色区域
    #   alpha = 0.5：折中
    #
    # 返回值：
    #   new_camera_matrix：新的 3x3 内参矩阵
    #   roi：有效图像区域 (x, y, w, h)
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

    # ==================== 保存标定结果 ====================
    # 保存为 .npz 格式（NumPy 压缩数组），方便后续加载使用
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

    print(f"\n标定参数已保存到：{args.output}")

    print("\n误差参考：")
    print("  < 0.3 像素：很好")
    print("  0.3 ~ 0.8 像素：正常可用")
    print("  0.8 ~ 1.5 像素：勉强可用")
    print("  > 1.5 像素：建议重新采集标定图片")


def undistort_live(args):
    """
    实时畸变矫正。

    功能：
        读取标定参数，对摄像头画面进行实时畸变矫正，并上下对比显示
        原始图像和矫正图像。

    畸变矫正原理：
        畸变矫正的本质是图像重映射（remap）：对于输出图像中的每个像素 (u, v)，
        计算它在原始畸变图像中对应的坐标 (u', v')，然后用双线性插值取像素值。

        OpenCV 的 initUndistortRectifyMap() 会预计算这个映射表（map1, map2），
        之后每一帧只需要 cv2.remap() 做查表和插值，速度很快。

    关于 dist_scale 参数：
        可以调节畸变矫正的强度。例如：
        - dist_scale = 1.0：完全按照标定结果矫正
        - dist_scale = 0.5：矫正强度减半（适合轻微畸变或不想过度矫正）
        - dist_scale = 0.0：不矫正（等于原始画面）
        只缩放径向畸变系数 k1, k2, k3，不缩放切向畸变 p1, p2。

    按键说明：
        q / Esc — 退出
        s — 保存当前矫正图像
        o — 切换上下对比显示 / 只显示矫正图

    参数：
        args (argparse.Namespace)：命令行参数
    """
    # ==================== 加载标定参数 ====================
    if not os.path.exists(args.calib_file):
        raise RuntimeError(f"找不到标定文件：{args.calib_file}")

    data = np.load(args.calib_file)

    camera_matrix = data["camera_matrix"]
    dist_coeffs_raw = data["dist_coeffs"]
    saved_image_size = tuple(data["image_size"].astype(int))

    # ==================== 畸变系数缩放 ====================
    # 保存畸变系数的原始形状，后面要还原给 OpenCV
    # OpenCV 的 dist_coeffs 可能是 (1,5)、(5,1)、(1,8)、(8,1) 等格式
    dist_shape = dist_coeffs_raw.shape

    # 拉平成一维，方便逐元素操作
    dist_flat = dist_coeffs_raw.reshape(-1).copy()

    # 只缩放径向畸变系数 k1, k2, k3
    # OpenCV 畸变系数的常见排列顺序：[k1, k2, p1, p2, k3]
    # 不缩放切向畸变 p1, p2，因为切向畸变通常很小且与径向畸变独立
    if len(dist_flat) >= 1:
        dist_flat[0] *= args.dist_scale   # k1：一阶径向畸变

    if len(dist_flat) >= 2:
        dist_flat[1] *= args.dist_scale   # k2：二阶径向畸变

    if len(dist_flat) >= 5:
        dist_flat[4] *= args.dist_scale   # k3：三阶径向畸变

    # 还原成原始形状
    dist_coeffs = dist_flat.reshape(dist_shape)

    print(f"\nDistortion scale: {args.dist_scale}")
    print("Original dist coeffs:")
    print(dist_coeffs_raw)
    print("Scaled dist coeffs:")
    print(dist_coeffs)

    print("读取标定参数：")
    print(f"  calib_file: {args.calib_file}")
    print(f"  saved image size: {saved_image_size}")

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nDist coeffs:")
    print(dist_coeffs)

    # ==================== 打开摄像头 ====================
    cap = open_usb_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_mjpg=not args.no_mjpg
    )

    current_image_size = (args.width, args.height)

    # 检查当前分辨率是否与标定分辨率一致
    if current_image_size != saved_image_size:
        print("\n警告：当前分辨率和标定分辨率不一致！")
        print(f"  标定分辨率：{saved_image_size}")
        print(f"  当前分辨率：{current_image_size}")
        print("  建议用相同分辨率进行标定和畸变矫正。\n")

    # ==================== 预计算映射表 ====================
    # getOptimalNewCameraMatrix 计算矫正后的新内参矩阵
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        current_image_size,
        args.alpha,
        current_image_size
    )

    # initUndistortRectifyMap 预计算畸变矫正的像素映射表
    # 这个映射表定义了：输出图像中每个像素 (u, v) 对应原始图像中的哪个位置
    #
    # map1, map2 的含义：
    #   map1[y][x] = 原始图像中对应的 x 坐标（浮点）
    #   map2[y][x] = 原始图像中对应的 y 坐标（浮点）
    #
    # CV_16SC2 表示 map1 使用 16 位有符号整数，2 通道（存储 x, y 坐标对）
    # 这比 CV_32FC2 更省内存，精度足够
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,                  # 不做额外旋转（R=None 表示单位矩阵）
        new_camera_matrix,
        current_image_size,
        cv2.CV_16SC2
    )

    print("\n开始实时畸变矫正")
    print("  q / Esc：退出")
    print("  s：保存当前矫正图像")
    print("  o：切换上下对比显示 / 只显示矫正图")
    print("  如果按键显示在终端里也没关系，本函数支持终端按键\n")

    # 初始化 FPS 计算器和保存计数器
    fps_counter = FPSCounter()
    save_count = len(glob.glob(os.path.join(USB_UNDISTORT_SAVE_DIR, "undistorted_usb_*.jpg")))
    show_original = args.show_original

    with TerminalKeyReader() as key_reader:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("警告：读取摄像头帧失败")
                continue

            # ==================== 畸变矫正 ====================
            # cv2.remap 使用预计算的映射表对图像做重映射
            # INTER_LINEAR：双线性插值（速度和质量的平衡）
            undistorted = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR
            )

            # 可选：裁掉矫正后的黑边区域
            if args.crop:
                x, y, w, h = roi
                undistorted = undistorted[y:y + h, x:x + w]

            # 计算 FPS
            show_fps = fps_counter.update()

            # ==================== 画面绘制 ====================
            if not args.headless:
                if show_original:
                    # 上下对比模式：上面原图，下面矫正图
                    original = frame.copy()

                    # 如果原图和矫正图尺寸不同（裁剪后），调整原图尺寸
                    if original.shape[:2] != undistorted.shape[:2]:
                        original = cv2.resize(
                            original,
                            (undistorted.shape[1], undistorted.shape[0])
                        )

                    original_show = original.copy()
                    undistorted_show = undistorted.copy()

                    cv2.putText(
                        original_show,
                        f"Original | FPS: {show_fps:.1f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),  # 黄色
                        2
                    )

                    cv2.putText(
                        undistorted_show,
                        "Undistorted | q/Esc quit | s save | o toggle",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

                    # 白色分隔线（8 像素高）
                    separator = np.full(
                        (8, original_show.shape[1], 3),
                        255,
                        dtype=np.uint8
                    )

                    # 上下拼接：原图 + 分隔线 + 矫正图
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

                    cv2.imshow("original(top) | undistorted(bottom)", combined)

                else:
                    # 只显示矫正图
                    display = undistorted.copy()

                    cv2.putText(
                        display,
                        f"Undistorted | FPS: {show_fps:.1f} | q/Esc quit | s save | o toggle",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),  # 绿色
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

                    cv2.imshow("undistorted", display)

            # ==================== 按键处理 ====================
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

            # q / Q / Esc：退出
            if key_char in ("q", "Q") or cv_key == 27:
                print("退出实时矫正")
                break

            # s / S：保存当前矫正图像
            if key_char in ("s", "S"):
                save_dir = USB_UNDISTORT_SAVE_DIR
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"undistorted_usb_{save_count:03d}.jpg")
                cv2.imwrite(filename, undistorted)
                print(f"[SAVE] {filename}")
                save_count += 1
                continue

            # o / O：切换对比显示模式
            if key_char in ("o", "O"):
                show_original = not show_original
                print(f"show_original = {show_original}")
                continue

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    """
    解析命令行参数。

    参数说明：
        --mode：运行模式（必须指定）
            capture   — 采集标定图片
            calibrate — 计算标定参数
            undistort  — 实时畸变矫正

        --device：摄像头设备节点（默认 /dev/video41）
        --width / --height：采集分辨率（默认 1920x1080）
        --fps：采集帧率（默认 30）
        --no-mjpg：禁用 MJPG，使用摄像头默认格式

        --board-cols / --board-rows：棋盘格内角点数量（默认 11x8）
        --square-size：方格边长 mm（默认 3.0）

        --output-dir：标定图片保存目录（默认 calib_images/）
        --input-dir：标定图片读取目录（默认 calib_images/）
        --output：标定结果输出文件（默认 camera_calib.npz）
        --calib-file：实时矫正使用的标定文件（默认 camera_calib.npz）

        --alpha：矫正视野参数（0=裁黑边，1=保留全部视野）
        --dist-scale：畸变矫正强度（1.0=原始标定结果）
        --crop：矫正后是否裁掉黑边
        --show-original：实时矫正时显示原图对比
        --display-scale：显示窗口缩放比例
        --detect-every：采集预览每隔 N 帧检测角点
        --detect-scale：采集预览角点检测缩放比例

    返回：
        argparse.Namespace：解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="USB camera chessboard calibration and undistortion"
    )

    # 运行模式
    parser.add_argument(
        "--mode",
        required=True,
        choices=["capture", "calibrate", "undistort"],
        help="capture=采集标定图片, calibrate=计算标定参数, undistort=实时畸变矫正"
    )

    # 摄像头参数
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="摄像头设备节点，例如 /dev/video41"
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

    # 标定板参数
    parser.add_argument(
        "--board-cols",
        type=int,
        default=DEFAULT_BOARD_COLS,
        help="棋盘格横向内角点数量。你的 12x9 方格应填 11。"
    )

    parser.add_argument(
        "--board-rows",
        type=int,
        default=DEFAULT_BOARD_ROWS,
        help="棋盘格纵向内角点数量。你的 12x9 方格应填 8。"
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE,
        help="方格边长，单位 mm。你的标定板是 3mm。"
    )

    # 输入输出路径
    parser.add_argument(
        "--output-dir",
        default="calib_images",
        help="采集标定图片保存目录"
    )

    parser.add_argument(
        "--input-dir",
        default="calib_images",
        help="标定图片输入目录"
    )

    parser.add_argument(
        "--output",
        default="camera_calib.npz",
        help="标定结果输出文件"
    )

    parser.add_argument(
        "--calib-file",
        default="camera_calib.npz",
        help="实时矫正使用的标定文件"
    )

    # 矫正参数
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

    # 显示参数
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
        help="无头模式，不显示 OpenCV 窗口（适用于无显示器的嵌入式设备）"
    )

    # 采集预览参数
    parser.add_argument(
        "--detect-every",
        type=int,
        default=5,
        help="每隔多少帧检测一次棋盘格角点。值越大，显示越流畅，但角点更新越慢。"
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

    根据 --mode 参数分发到对应的处理函数：
        capture   → capture_images()：采集标定图片
        calibrate → calibrate_camera()：计算标定参数
        undistort  → undistort_live()：实时畸变矫正
    """
    args = parse_args()

    print("当前参数：")
    print(f"  mode        : {args.mode}")
    print(f"  device      : {args.device}")
    print(f"  size        : {args.width}x{args.height}")
    print(f"  fps         : {args.fps}")
    print(f"  board       : {args.board_cols} x {args.board_rows} inner corners")
    print(f"  square size : {args.square_size} mm")

    # 检查 DISPLAY 环境变量，提示用户设置
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
