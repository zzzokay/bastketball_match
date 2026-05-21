#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
双目摄像头物理安装对齐辅助工具

功能：
    实时显示两个摄像头的画面（支持畸变矫正），并叠加水平参考线和角度信息，
    帮助你调整摄像头的安装角度和高度，使两个摄像头达到水平对齐。

使用方法：
    1. 打开程序，同时显示左右摄像头画面
    2. 在画面中放置一个水平参考物（如尺子、书本边缘）
    3. 根据叠加的参考线调整摄像头角度
    4. 当左右画面中的参考线重合时，说明对齐完成

按键说明：
    r — 重置参考线位置
    ↑/↓ — 调整参考线位置
    s — 保存当前左右画面（畸变矫正后）
    q / Esc — 退出
"""

import os
import sys
import select
import termios
import tty
import argparse

import cv2
import numpy as np


# 默认摄像头设备
# Left = OV13588 CSI 摄像头 = /dev/video22
# Right = USB 摄像头 = /dev/video41
# DEFAULT_LEFT_DEVICE = "/dev/video22"
DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"

# 保存目录
DEFAULT_LEFT_DIR = "/home/elf/work/basketball/stereo_undistorted_calib/Left"
DEFAULT_RIGHT_DIR = "/home/elf/work/basketball/stereo_undistorted_calib/Right"

# 标定文件
# DEFAULT_LEFT_CALIB = "/home/elf/work/basketball/camera_calib_ov13588.npz"
DEFAULT_LEFT_CALIB = "/home/elf/work/basketball/camera_usb2_calib.npz"
DEFAULT_RIGHT_CALIB = "/home/elf/work/basketball/camera_calib.npz"


class TerminalKeyReader:
    """终端按键读取器"""

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
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1)
        return None


def open_camera_gst(device, width, height, fps, pixel_format="NV12"):
    """使用 GStreamer 打开摄像头（用于 OV13588 等 CSI 摄像头）"""
    pixel_format = pixel_format.upper()
    pipeline = (
        f"v4l2src device={device} ! "
        f"video/x-raw,format={pixel_format},width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink"
    )

    print(f"尝试使用 GStreamer 打开摄像头：")
    print(f"  {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"GStreamer 打开摄像头失败：{device}")

    print(f"GStreamer 摄像头已打开：{device}")
    return cap


def open_camera_v4l2(device, width, height, fps, use_mjpg=True):
    """使用 V4L2 打开摄像头（用于 USB 摄像头）"""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"V4L2 打开摄像头失败：{device}")

    if use_mjpg:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"V4L2 摄像头已打开：{device} ({real_width}x{real_height})")

    return cap


def draw_alignment_guide(frame, guide_y, label=""):
    """
    在画面上绘制对齐辅助线

    参数：
        frame: 图像
        guide_y: 水平参考线的 y 坐标
        label: 标签文字
    """
    h, w = frame.shape[:2]

    # 绘制水平参考线（绿色）
    cv2.line(frame, (0, guide_y), (w, guide_y), (0, 255, 0), 2)

    # 绘制垂直中心线（黄色虚线效果）
    center_x = w // 2
    for y in range(0, h, 20):
        cv2.line(frame, (center_x, y), (center_x, min(y + 10, h)), (0, 255, 255), 1)

    # 绘制十字准星
    cross_size = 30
    cv2.line(frame, (center_x - cross_size, guide_y),
             (center_x + cross_size, guide_y), (0, 255, 0), 3)
    cv2.line(frame, (center_x, guide_y - cross_size),
             (center_x, guide_y + cross_size), (0, 255, 0), 3)

    # 显示坐标信息
    cv2.putText(
        frame,
        f"Guide Y: {guide_y} | {label}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return frame


def calculate_tilt_angle(frame, guide_y):
    """
    简单估算画面中的水平倾斜角度

    通过检测画面中的水平线条来估算倾斜
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return 0.0, 0

    # 找出接近水平的线（角度在 ±15 度以内）
    horizontal_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > 50:  # 水平方向长度要够
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:  # 接近水平
                horizontal_angles.append(angle)

    if not horizontal_angles:
        return 0.0, 0

    avg_angle = np.mean(horizontal_angles)
    return avg_angle, len(horizontal_angles)


def main():
    parser = argparse.ArgumentParser(description="双目摄像头对齐辅助工具")
    parser.add_argument("--left", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备")
    parser.add_argument("--right", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备")
    parser.add_argument("--width", type=int, default=1920, help="采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="采集高度")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率")
    parser.add_argument("--display-scale", type=float, default=0.2, help="显示缩放比例")
    parser.add_argument("--left-calib", default=DEFAULT_LEFT_CALIB, help="左摄像头标定文件")
    parser.add_argument("--right-calib", default=DEFAULT_RIGHT_CALIB, help="右摄像头标定文件")
    parser.add_argument("--no-undistort", action="store_true", help="不进行畸变矫正")
    parser.add_argument("--alpha", type=float, default=0.0, help="矫正视野参数。0=裁黑边，1=保留全部视野")
    parser.add_argument("--dist-scale", type=float, default=1.0, help="畸变矫正强度。1.0=原始标定结果")
    parser.add_argument("--left-format", choices=["nv12", "uyvy", "mjpg"], default="nv12",
                        help="左摄像头像素格式（OV13588 用 nv12）")
    parser.add_argument("--right-format", choices=["nv12", "uyvy", "mjpg"], default="mjpg",
                        help="右摄像头像素格式（USB 摄像头用 mjpg）")
    args = parser.parse_args()

    print("=" * 60)
    print("双目摄像头对齐辅助工具")
    print("=" * 60)
    print()
    print("使用方法：")
    print("  1. 在画面前方放置一个水平参考物（尺子、书本边缘等）")
    print("  2. 调整摄像头角度，使参考物与绿色参考线重合")
    print("  3. 当左右画面的参考线位置一致时，说明对齐完成")
    print()
    print("按键说明：")
    print("  r — 重置参考线到画面中心")
    print("  ↑/↓ — 上下移动参考线")
    print("  a/A — 调整左摄像头角度提示")
    print("  d/D — 调整右摄像头角度提示")
    print("  s — 保存当前左右画面（畸变矫正后）")
    print("  q / Esc — 退出")
    print()

    # 打开左右摄像头（根据格式选择打开方式，GStreamer 失败则 fallback 到 V4L2）
    def open_camera_smart(device, width, height, fps, pixel_format):
        if pixel_format in ("nv12", "uyvy"):
            try:
                return open_camera_gst(device, width, height, fps, pixel_format)
            except Exception as e:
                print(f"GStreamer 打开失败，尝试 V4L2：{e}")
        return open_camera_v4l2(device, width, height, fps, use_mjpg=True)

    cap_left = open_camera_smart(args.left, args.width, args.height, args.fps, args.left_format)
    if cap_left is None:
        return

    cap_right = open_camera_smart(args.right, args.width, args.height, args.fps, args.right_format)
    if cap_right is None:
        cap_left.release()
        return

    # 预热摄像头（OV13588 需要几帧才能稳定）
    print("预热摄像头...")
    for _ in range(10):
        cap_left.read()
        cap_right.read()

    # ==================== 加载标定参数并预计算畸变矫正映射表 ====================
    map_left_1, map_left_2 = None, None
    map_right_1, map_right_2 = None, None
    pixel_aspect_left = 1.0
    pixel_aspect_right = 1.0

    if not args.no_undistort:
        image_size = (args.width, args.height)

        # 加载左摄像头标定参数
        if os.path.exists(args.left_calib):
            print(f"加载左摄像头标定文件: {args.left_calib}")
            data_left = np.load(args.left_calib)
            K_left = data_left["camera_matrix"]
            D_left_raw = data_left["dist_coeffs"]

            # 应用 dist_scale 缩放畸变系数
            D_left_flat = D_left_raw.reshape(-1).copy()
            if len(D_left_flat) >= 1:
                D_left_flat[0] *= args.dist_scale
            if len(D_left_flat) >= 2:
                D_left_flat[1] *= args.dist_scale
            if len(D_left_flat) >= 5:
                D_left_flat[4] *= args.dist_scale
            D_left = D_left_flat.reshape(D_left_raw.shape)

            new_K_left, roi_left = cv2.getOptimalNewCameraMatrix(
                K_left, D_left, image_size, args.alpha, image_size
            )
            map_left_1, map_left_2 = cv2.initUndistortRectifyMap(
                K_left, D_left, None, new_K_left, image_size, cv2.CV_16SC2
            )

            # 检测 anamorphic scaling（像素宽高比）
            pixel_aspect_left = K_left[1, 1] / K_left[0, 0]
            print(f"  K_left:\n{K_left}")
            print(f"  D_left: {D_left.reshape(-1)}")
            print(f"  pixel_aspect_left: {pixel_aspect_left:.4f}")
        else:
            print(f"警告：找不到左摄像头标定文件 {args.left_calib}")

        # 加载右摄像头标定参数
        if os.path.exists(args.right_calib):
            print(f"加载右摄像头标定文件: {args.right_calib}")
            data_right = np.load(args.right_calib)
            K_right = data_right["camera_matrix"]
            D_right_raw = data_right["dist_coeffs"]

            # 应用 dist_scale 缩放畸变系数
            D_right_flat = D_right_raw.reshape(-1).copy()
            if len(D_right_flat) >= 1:
                D_right_flat[0] *= args.dist_scale
            if len(D_right_flat) >= 2:
                D_right_flat[1] *= args.dist_scale
            if len(D_right_flat) >= 5:
                D_right_flat[4] *= args.dist_scale
            D_right = D_right_flat.reshape(D_right_raw.shape)

            new_K_right, roi_right = cv2.getOptimalNewCameraMatrix(
                K_right, D_right, image_size, args.alpha, image_size
            )
            map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
                K_right, D_right, None, new_K_right, image_size, cv2.CV_16SC2
            )

            # 检测 anamorphic scaling
            pixel_aspect_right = K_right[1, 1] / K_right[0, 0]
            print(f"  K_right:\n{K_right}")
            print(f"  D_right: {D_right.reshape(-1)}")
            print(f"  pixel_aspect_right: {pixel_aspect_right:.4f}")
        else:
            print(f"警告：找不到右摄像头标定文件 {args.right_calib}")

        if map_left_1 is not None and map_right_1 is not None:
            print(f"畸变矫正已启用 (alpha={args.alpha}, dist_scale={args.dist_scale})")
        else:
            print("警告：标定文件不完整，将使用原始画面")
    else:
        print("畸变矫正已禁用（--no-undistort）")

    # 创建保存目录
    os.makedirs(DEFAULT_LEFT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RIGHT_DIR, exist_ok=True)

    # 保存计数器
    save_count = 0

    # 参考线初始位置（画面中心）
    guide_y_left = args.height // 2
    guide_y_right = args.height // 2

    # 角度调整提示偏移
    angle_offset_left = 0
    angle_offset_right = 0

    with TerminalKeyReader() as key_reader:
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                print("警告：读取帧失败")
                continue

            # ==================== 畸变矫正 ====================
            frame_left_undistorted = frame_left
            frame_right_undistorted = frame_right

            if map_left_1 is not None and map_left_2 is not None:
                frame_left_undistorted = cv2.remap(
                    frame_left, map_left_1, map_left_2, cv2.INTER_LINEAR
                )
                # Anamorphic fix: 还原正方形像素
                if abs(pixel_aspect_left - 1.0) > 0.01:
                    h, w = frame_left_undistorted.shape[:2]
                    new_w = int(w * pixel_aspect_left)
                    frame_left_undistorted = cv2.resize(
                        frame_left_undistorted, (new_w, h),
                        interpolation=cv2.INTER_LINEAR
                    )

            if map_right_1 is not None and map_right_2 is not None:
                frame_right_undistorted = cv2.remap(
                    frame_right, map_right_1, map_right_2, cv2.INTER_LINEAR
                )
                # Anamorphic fix: 还原正方形像素
                if abs(pixel_aspect_right - 1.0) > 0.01:
                    h, w = frame_right_undistorted.shape[:2]
                    new_w = int(w * pixel_aspect_right)
                    frame_right_undistorted = cv2.resize(
                        frame_right_undistorted, (new_w, h),
                        interpolation=cv2.INTER_LINEAR
                    )

            # 计算倾斜角度（使用矫正后的画面）
            tilt_left, lines_left = calculate_tilt_angle(frame_left_undistorted, guide_y_left)
            tilt_right, lines_right = calculate_tilt_angle(frame_right_undistorted, guide_y_right)

            # 绘制对齐辅助线（使用矫正后的画面）
            display_left = frame_left_undistorted.copy()
            display_right = frame_right_undistorted.copy()

            # 添加角度提示
            adjusted_y_left = guide_y_left + angle_offset_left
            adjusted_y_right = guide_y_right + angle_offset_right

            display_left = draw_alignment_guide(
                display_left,
                adjusted_y_left,
                f"Tilt: {tilt_left:.1f}° | Lines: {lines_left}"
            )

            display_right = draw_alignment_guide(
                display_right,
                adjusted_y_right,
                f"Tilt: {tilt_right:.1f}° | Lines: {lines_right}"
            )

            # 添加左右标签
            cv2.putText(display_left, "LEFT", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_right, "RIGHT", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 显示角度差异提示
            angle_diff = tilt_left - tilt_right
            diff_color = (0, 255, 0) if abs(angle_diff) < 2 else (0, 0, 255)
            diff_text = f"Angle Diff: {angle_diff:.1f}° | Saved: {save_count}"

            # 缩放显示
            if args.display_scale != 1.0:
                display_left = cv2.resize(display_left, None,
                                          fx=args.display_scale,
                                          fy=args.display_scale,
                                          interpolation=cv2.INTER_AREA)
                display_right = cv2.resize(display_right, None,
                                           fx=args.display_scale,
                                           fy=args.display_scale,
                                           interpolation=cv2.INTER_AREA)

            # 左右拼接显示
            separator = np.full((display_left.shape[0], 5, 3), 255, dtype=np.uint8)
            combined = np.hstack((display_left, separator, display_right))

            # 在顶部添加状态栏
            status_bar = np.full((50, combined.shape[1], 3), 40, dtype=np.uint8)
            cv2.putText(status_bar, diff_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, diff_color, 2)
            cv2.putText(status_bar, "s: save | r: reset | ↑↓: move | q: quit",
                        (combined.shape[1] - 420, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            combined = np.vstack((status_bar, combined))

            cv2.imshow("Camera Alignment", combined)

            # 按键处理
            cv_key = cv2.waitKey(1)
            term_key = key_reader.read_key()

            key_char = None
            if cv_key != -1:
                key_char = chr(cv_key & 0xFF)
            if term_key is not None:
                key_char = term_key

            if key_char is None:
                continue

            # 退出
            if key_char in ("q", "Q") or cv_key == 27:
                print("退出")
                break

            # 保存左右图片（畸变矫正后）
            if key_char in ("s", "S"):
                left_path = os.path.join(DEFAULT_LEFT_DIR, f"align_{save_count:03d}.jpg")
                right_path = os.path.join(DEFAULT_RIGHT_DIR, f"align_{save_count:03d}.jpg")
                cv2.imwrite(left_path, frame_left_undistorted)
                cv2.imwrite(right_path, frame_right_undistorted)
                print(f"[SAVE] Left:  {left_path}")
                print(f"[SAVE] Right: {right_path}")
                save_count += 1
                continue

            # 重置参考线
            if key_char in ("r", "R"):
                guide_y_left = args.height // 2
                guide_y_right = args.height // 2
                angle_offset_left = 0
                angle_offset_right = 0
                print("参考线已重置")

            # 上移参考线
            if cv_key == 82 or term_key == '\x1b[A':  # Up arrow
                guide_y_left = max(50, guide_y_left - 20)
                guide_y_right = max(50, guide_y_right - 20)
                print(f"参考线上移: {guide_y_left}")

            # 下移参考线
            if cv_key == 84 or term_key == '\x1b[B':  # Down arrow
                guide_y_left = min(args.height - 50, guide_y_left + 20)
                guide_y_right = min(args.height - 50, guide_y_right + 20)
                print(f"参考线下移: {guide_y_left}")

            # 调整左摄像头角度提示
            if key_char in ("a",):
                angle_offset_left -= 10
                print(f"左摄像头角度偏移: {angle_offset_left}")
            if key_char in ("A",):
                angle_offset_left += 10
                print(f"左摄像头角度偏移: {angle_offset_left}")

            # 调整右摄像头角度提示
            if key_char in ("d",):
                angle_offset_right -= 10
                print(f"右摄像头角度偏移: {angle_offset_right}")
            if key_char in ("D",):
                angle_offset_right += 10
                print(f"右摄像头角度偏移: {angle_offset_right}")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
