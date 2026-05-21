#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RK3588 + OV13588 摄像头标定程序

默认采集链路：
    v4l2src device=/dev/video22 !
    video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 !
    videoconvert !
    video/x-raw,format=BGR !
    appsink

使用方式：
    1. 采集标定图：
       python3 calib_ov13588_camera.py --mode capture

    2. 计算标定参数：
       python3 calib_ov13588_camera.py --mode calibrate

    3. 实时畸变矫正：
       python3 calib_ov13588_camera.py --mode undistort

按键：
    c：保存当前检测到棋盘格角点的标定图
    s：直接保存当前原图
    q / Esc：退出
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


# ============================================================
# 默认参数
# ============================================================

DEFAULT_DEVICE = "/dev/video22"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
# DEFAULT_WIDTH = 2160
# DEFAULT_HEIGHT = 3840
DEFAULT_FPS = 30

# 12 x 9 方格，对应 11 x 8 内角点
DEFAULT_BOARD_COLS = 11
DEFAULT_BOARD_ROWS = 8
DEFAULT_SQUARE_SIZE = 22.0  # mm

DEFAULT_OUTPUT_DIR = "calib_images_ov13588"
DEFAULT_CALIB_FILE = "camera_calib_ov13588.npz"


# ============================================================
# 棋盘格检测参数
# ============================================================

CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE
)

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


# ============================================================
# 工具类
# ============================================================

class FPSCounter:
    def __init__(self):
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def update(self):
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_time

        if dt >= 1.0:
            self.fps = self.frame_count / dt
            self.frame_count = 0
            self.last_time = now

        return self.fps


class TerminalKeyReader:
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


# ============================================================
# 摄像头打开与格式转换
# ============================================================

def fourcc_to_str(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def make_gst_pipeline(device, width, height, fps, pixel_format):
    pixel_format = pixel_format.upper()

    pipeline = (
        f"v4l2src device={device} ! "
        f"video/x-raw,format={pixel_format},width={width},height={height},framerate={fps}/1 ! "
        f"queue leaky=downstream max-size-buffers=2 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=2 sync=false"
    )

    return pipeline


def open_camera_gst(args):
    pipeline = make_gst_pipeline(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixel_format=args.format
    )

    print("\n尝试使用 GStreamer 打开摄像头：")
    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("GStreamer 打开摄像头失败")

    print("GStreamer 摄像头已打开")
    return cap


def open_camera_v4l2(args):
    print("\n尝试使用 OpenCV V4L2 打开摄像头")

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"V4L2 打开摄像头失败：{args.device}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)

    if args.format == "nv12":
        fourcc = cv2.VideoWriter_fourcc(*"NV12")
    elif args.format == "uyvy":
        fourcc = cv2.VideoWriter_fourcc(*"UYVY")
    else:
        raise RuntimeError(f"不支持的格式：{args.format}")

    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # 不关闭自动转换。
    # 很多 OpenCV V4L2 后端会把 NV12 自动转成 BGR 返回。
    # 后面 ensure_bgr() 会自动判断，不会再强行 reshape。
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    real_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    print("V4L2 摄像头已打开：")
    print(f"  device : {args.device}")
    print(f"  size   : {real_w}x{real_h}")
    print(f"  fps    : {real_fps}")
    print(f"  fourcc : {real_fourcc}")

    return cap


def open_camera(args):
    """
    backend:
        auto：优先 GStreamer，失败后 V4L2
        gst ：只用 GStreamer
        v4l2：只用 V4L2
    """

    if args.backend == "gst":
        return open_camera_gst(args)

    if args.backend == "v4l2":
        return open_camera_v4l2(args)

    # auto
    try:
        return open_camera_gst(args)
    except Exception as e:
        print(f"\nGStreamer 打开失败，准备尝试 V4L2：{e}")
        return open_camera_v4l2(args)


def ensure_bgr(frame, width, height):
    """
    把 cap.read() 返回的数据统一变成 BGR。

    兼容情况：
        1. GStreamer appsink 已经返回 BGR：H x W x 3
        2. OpenCV V4L2 已经自动返回 BGR：H x W x 3
        3. OpenCV 返回 BGRA：H x W x 4
        4. OpenCV 返回原始 NV12：H*3/2 x W 或一维 buffer
        5. OpenCV 返回 UYVY：H x W x 2
        6. OpenCV 返回灰度：H x W
    """

    if frame is None:
        raise RuntimeError("frame is None")

    # BGR
    # OpenCV V4L2 在部分 RK 平台上虽然返回 3 通道，
    # 但实际顺序可能是 RGB。cv2.imshow / cv2.imwrite 需要 BGR。
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # BGRA
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # UYVY: H x W x 2
    if frame.ndim == 3 and frame.shape[2] == 2:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)

    # NV12 原始数据：width * height * 3 / 2
    expected_nv12_size = width * height * 3 // 2
    if frame.size == expected_nv12_size:
        yuv = frame.reshape(height * 3 // 2, width)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    # 灰度
    if frame.ndim == 2 and frame.shape[0] == height and frame.shape[1] == width:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(
        "未知帧格式："
        f"shape={frame.shape}, dtype={frame.dtype}, size={frame.size}, "
        f"expected_nv12_size={expected_nv12_size}, "
        f"expected_bgr_size={width * height * 3}"
    )


def read_bgr_frame(cap, args, debug=False):
    ret, raw = cap.read()

    if not ret:
        return False, None

    if debug:
        print("第一帧信息：")
        print(f"  raw.shape = {raw.shape}")
        print(f"  raw.dtype = {raw.dtype}")
        print(f"  raw.size  = {raw.size}")

    frame = ensure_bgr(raw, args.width, args.height)
    return True, frame


def warmup_camera(cap, args, n=10):
    for i in range(n):
        ret, _ = read_bgr_frame(cap, args, debug=(i == 0))
        if not ret:
            time.sleep(0.05)


# ============================================================
# 标定相关
# ============================================================

def create_object_points(board_cols, board_rows, square_size):
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def find_corners(frame, board_cols, board_rows, detect_scale=1.0):
    if frame is None:
        return False, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detect_scale <= 0 or detect_scale > 1.0:
        detect_scale = 1.0

    if detect_scale != 1.0:
        small = cv2.resize(
            frame,
            None,
            fx=detect_scale,
            fy=detect_scale,
            interpolation=cv2.INTER_AREA
        )
        detect_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
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
    used_pattern = None

    for pattern_size in patterns:
        # 优先使用新版 SB 棋盘格检测，更适合高分辨率/畸变/光照变化
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
                used_pattern = pattern_size
                break

        # 回退到老版检测
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        found, corners = cv2.findChessboardCorners(
            detect_gray_eq,
            pattern_size,
            flags
        )

        if found:
            used_pattern = pattern_size
            break

    if not found or corners is None:
        return False, None, gray

    # 老版 findChessboardCorners 需要 cornerSubPix
    # SB 返回的角点已经比较精细，但再 refine 一次通常也没问题
    try:
        corners = cv2.cornerSubPix(
            detect_gray,
            corners,
            winSize=SUBPIX_WIN_SIZE_PREVIEW if detect_scale != 1.0 else SUBPIX_WIN_SIZE_FULL,
            zeroZone=(-1, -1),
            criteria=SUBPIX_CRITERIA_PREVIEW if detect_scale != 1.0 else SUBPIX_CRITERIA_FULL
        )
    except cv2.error:
        pass

    if detect_scale != 1.0:
        corners = corners / detect_scale

    # 如果实际用的是反过来的 8x11，drawChessboardCorners 时需要对应 pattern。
    # 你当前外部代码固定传 11x8，所以这里先只返回 corners。
    return True, corners, gray


def next_save_index(output_dir):
    nums = []

    for pattern in ("calib_*.jpg", "preview_*.jpg", "raw_*.jpg", "undistorted_*.jpg"):
        for path in glob.glob(os.path.join(output_dir, pattern)):
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
            try:
                nums.append(int(stem.split("_")[-1]))
            except Exception:
                pass

    if not nums:
        return 0

    return max(nums) + 1


# ============================================================
# mode: capture
# ============================================================

def capture_images(args):
    os.makedirs(args.output_dir, exist_ok=True)

    cap = open_camera(args)
    warmup_camera(cap, args, n=args.warmup)

    save_index = next_save_index(args.output_dir)
    valid_count = len(glob.glob(os.path.join(args.output_dir, "calib_*.jpg")))

    print("\n采集说明：")
    print("  c    : 保存当前检测成功的棋盘格标定图")
    print("  s    : 直接保存当前原图")
    print("  q/Esc: 退出")
    print(f"  输出目录：{args.output_dir}")
    print(f"  当前已有有效标定图：{valid_count}")
    print("\n标定板参数：")
    print(f"  内角点：{args.board_cols} x {args.board_rows}")
    print(f"  方格边长：{args.square_size} mm")
    print("\n摄像头参数：")
    print(f"  backend : {args.backend}")
    print(f"  device  : {args.device}")
    print(f"  format  : {args.format}")
    print(f"  size    : {args.width}x{args.height}")
    print(f"  fps     : {args.fps}\n")

    fps_counter = FPSCounter()
    frame_index = 0

    last_frame = None
    last_found = False
    last_corners = None

    with TerminalKeyReader() as key_reader:
        while True:
            ret, frame = read_bgr_frame(cap, args)

            if not ret:
                print("警告：读取摄像头帧失败")
                time.sleep(0.01)
                continue

            last_frame = frame.copy()
            frame_index += 1
            fps = fps_counter.update()

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
                    status_text = "Corners NOT found"
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
                    f"Valid saved: {valid_count} | FPS: {fps:.1f}",
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
                    display = cv2.resize(
                        display,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA
                    )

                cv2.imshow("OV13588 calibration capture", display)

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
                    f"raw_{save_index:03d}.jpg"
                )
                cv2.imwrite(filename, last_frame)
                print(f"[RAW SAVE] {filename}")
                save_index += 1
                continue

            if key_char in ("c", "C"):
                if last_frame is None:
                    print("[FAIL] 当前没有图像")
                    continue

                print("正在全分辨率重新检测角点...")

                found_full, corners_full, _ = find_corners(
                    last_frame,
                    args.board_cols,
                    args.board_rows,
                    detect_scale=1.0
                )

                if not found_full:
                    fail_filename = os.path.join(
                        args.output_dir,
                        f"fail_{save_index:03d}.jpg"
                    )
                    cv2.imwrite(fail_filename, last_frame)
                    print("[FAIL] 全分辨率未检测到棋盘格角点，未保存标定图")
                    print(f"[DEBUG] 已保存失败帧用于排查：{fail_filename}")
                    save_index += 1
                    continue

                calib_filename = os.path.join(
                    args.output_dir,
                    f"calib_{save_index:03d}.jpg"
                )

                preview_filename = os.path.join(
                    args.output_dir,
                    f"preview_{save_index:03d}.jpg"
                )

                cv2.imwrite(calib_filename, last_frame)

                preview = last_frame.copy()
                cv2.drawChessboardCorners(
                    preview,
                    (args.board_cols, args.board_rows),
                    corners_full,
                    found_full
                )
                cv2.imwrite(preview_filename, preview)

                valid_count += 1

                print(f"[OK] 保存标定图：{calib_filename}")
                print(f"[OK] 保存预览图：{preview_filename}")
                print(f"[OK] 当前有效标定图数量：{valid_count}")

                save_index += 1
                continue

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# mode: calibrate
# ============================================================

def calibrate_camera(args):
    image_paths = sorted(
        glob.glob(os.path.join(args.input_dir, "calib_*.jpg"))
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"没有找到标定图片：{args.input_dir}/calib_*.jpg")

    print(f"找到 {len(image_paths)} 张标定图片")

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
            print(f"[跳过] 无法读取：{path}")
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
            print(f"[失败] 未检测到棋盘格：{path}")

    valid_count = len(object_points)

    if valid_count < 8:
        raise RuntimeError(
            f"有效标定图太少：{valid_count} 张。建议至少 15~30 张。"
        )

    print("\n开始相机标定...")

    w, h = image_size

    init_camera_matrix = np.array(
        [
            [w, 0, w / 2.0],
            [0, w, h / 2.0],
            [0, 0, 1.0],
        ],
        dtype=np.float64
    )

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

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        args.alpha,
        image_size
    )

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

    print("\n================ 标定结果 ================")
    print(f"有效图片数量：{valid_count}")
    print(f"图像尺寸：{image_size}")
    print("\nRMS 误差：")
    print(rms)
    print("\n平均重投影误差：")
    print(mean_error)
    print("\nCamera Matrix：")
    print(camera_matrix)
    print("\nDist Coeffs：")
    print(dist_coeffs)
    print("\nNew Camera Matrix：")
    print(new_camera_matrix)
    print("\nROI：")
    print(roi)
    print(f"\n标定文件已保存：{args.output}")

    print("\n误差参考：")
    print("  < 0.3 像素：很好")
    print("  0.3 ~ 0.8 像素：正常可用")
    print("  0.8 ~ 1.5 像素：勉强可用")
    print("  > 1.5 像素：建议重新采集")


# ============================================================
# mode: undistort
# ============================================================

def undistort_live(args):
    if not os.path.exists(args.calib_file):
        raise RuntimeError(f"找不到标定文件：{args.calib_file}")

    data = np.load(args.calib_file)

    camera_matrix = data["camera_matrix"]
    dist_coeffs_raw = data["dist_coeffs"]
    saved_image_size = tuple(data["image_size"].astype(int))

    dist_shape = dist_coeffs_raw.shape
    dist_flat = dist_coeffs_raw.reshape(-1).copy()

    if len(dist_flat) >= 1:
        dist_flat[0] *= args.dist_scale
    if len(dist_flat) >= 2:
        dist_flat[1] *= args.dist_scale
    if len(dist_flat) >= 5:
        dist_flat[4] *= args.dist_scale

    dist_coeffs = dist_flat.reshape(dist_shape)

    current_image_size = (args.width, args.height)

    print("\n读取标定参数：")
    print(f"  calib_file       : {args.calib_file}")
    print(f"  saved image size : {saved_image_size}")
    print(f"  current size     : {current_image_size}")
    print(f"  dist_scale       : {args.dist_scale}")

    if current_image_size != saved_image_size:
        print("\n警告：当前分辨率和标定分辨率不一致，建议使用相同分辨率。\n")

    print("\nCamera Matrix：")
    print(camera_matrix)
    print("\nDist Coeffs：")
    print(dist_coeffs)

    cap = open_camera(args)
    warmup_camera(cap, args, n=args.warmup)

    # 检测 ISP 变形缩放（anamorphic scaling）
    # OV13855 传感器 4224x3136 (4:3)，ISP 输出 1920x1080 (16:9)
    # 导致像素横向比纵向宽约 1.32 倍，矫正后需要 resize 恢复正方形像素
    pixel_aspect = camera_matrix[1, 1] / camera_matrix[0, 0]  # fy/fx ≈ 0.756
    need_anamorphic_fix = abs(pixel_aspect - 1.0) > 0.01

    if need_anamorphic_fix:
        print(f"\n检测到变形像素（pixel aspect = {pixel_aspect:.4f}），矫正后将自动还原为正方形像素")

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

    print("\n开始实时畸变矫正：")
    print("  s    : 保存当前矫正图")
    print("  o    : 切换原图/矫正图对比显示")
    print("  q/Esc: 退出\n")

    fps_counter = FPSCounter()
    save_index = 0
    show_original = args.show_original

    with TerminalKeyReader() as key_reader:
        while True:
            ret, frame = read_bgr_frame(cap, args)

            if not ret:
                print("警告：读取摄像头帧失败")
                time.sleep(0.01)
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

            # 还原正方形像素：将变形图像的宽度乘以 pixel_aspect
            if need_anamorphic_fix:
                h, w = undistorted.shape[:2]
                new_w = int(w * pixel_aspect)
                undistorted = cv2.resize(undistorted, (new_w, h), interpolation=cv2.INTER_LINEAR)

            fps = fps_counter.update()

            if not args.headless:
                if show_original:
                    original_show = frame.copy()
                    undistorted_show = undistorted.copy()

                    if original_show.shape[:2] != undistorted_show.shape[:2]:
                        original_show = cv2.resize(
                            original_show,
                            (undistorted_show.shape[1], undistorted_show.shape[0])
                        )

                    cv2.putText(
                        original_show,
                        f"Original | FPS: {fps:.1f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        undistorted_show,
                        "Undistorted | s save | o toggle | q quit",
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

                    display = np.vstack(
                        [original_show, separator, undistorted_show]
                    )

                else:
                    display = undistorted.copy()

                    cv2.putText(
                        display,
                        f"Undistorted | FPS: {fps:.1f} | s save | o toggle | q quit",
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

                cv2.imshow("OV13588 undistort", display)

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
                print("退出实时矫正")
                break

            if key_char in ("s", "S"):
                save_dir = "/home/elf/work/basketball/stereo_undistorted_calib/Left"
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"undistorted_ov13588_{save_index:03d}.jpg")
                cv2.imwrite(filename, undistorted)
                print(f"[SAVE] {filename}")
                save_index += 1
                continue

            if key_char in ("o", "O"):
                show_original = not show_original
                print(f"show_original = {show_original}")
                continue

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RK3588 + OV13588 camera calibration"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["capture", "calibrate", "undistort"],
        help="capture=采集标定图, calibrate=计算标定参数, undistort=实时矫正"
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "gst", "v4l2"],
        default="auto",
        help="摄像头后端。默认 auto：优先 GStreamer，失败后 V4L2"
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="摄像头设备节点"
    )

    parser.add_argument(
        "--format",
        choices=["nv12", "uyvy"],
        default="nv12",
        help="摄像头像素格式。RK3588 + OV13588 建议使用 nv12"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="采集宽度"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="采集高度"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="采集帧率"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="打开摄像头后丢弃的预热帧数"
    )

    parser.add_argument(
        "--board-cols",
        type=int,
        default=DEFAULT_BOARD_COLS,
        help="棋盘格横向内角点数量"
    )

    parser.add_argument(
        "--board-rows",
        type=int,
        default=DEFAULT_BOARD_ROWS,
        help="棋盘格纵向内角点数量"
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE,
        help="棋盘格方格边长，单位 mm"
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="采集图片保存目录"
    )

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="标定图片输入目录"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_CALIB_FILE,
        help="标定结果输出 npz 文件"
    )

    parser.add_argument(
        "--calib-file",
        default=DEFAULT_CALIB_FILE,
        help="实时矫正使用的标定文件"
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
        default=1.0,
        help="畸变矫正强度。1.0=原始标定结果，0.8=减弱"
    )

    parser.add_argument(
        "--crop",
        action="store_true",
        help="矫正后裁剪掉黑边"
    )

    parser.add_argument(
        "--show-original",
        action="store_true",
        help="实时矫正时显示原图和矫正图对比"
    )

    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.5,
        help="显示缩放比例"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="无窗口模式，只通过终端按键控制"
    )

    parser.add_argument(
        "--detect-every",
        type=int,
        default=5,
        help="采集预览时每隔多少帧检测一次棋盘格"
    )

    parser.add_argument(
        "--detect-scale",
        type=float,
        default=0.5,
        help="采集预览检测棋盘格时的缩放比例"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\nRK3588 + OV13588 相机标定程序")
    print("当前参数：")
    print(f"  mode        : {args.mode}")
    print(f"  backend     : {args.backend}")
    print(f"  device      : {args.device}")
    print(f"  format      : {args.format}")
    print(f"  size        : {args.width}x{args.height}")
    print(f"  fps         : {args.fps}")
    print(f"  board       : {args.board_cols} x {args.board_rows} inner corners")
    print(f"  square size : {args.square_size} mm")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n提示：未检测到 DISPLAY 环境变量，OpenCV 窗口可能无法显示。")
        print("可以先执行：export DISPLAY=:0\n")

    if args.mode == "capture":
        capture_images(args)
    elif args.mode == "calibrate":
        calibrate_camera(args)
    elif args.mode == "undistort":
        undistort_live(args)


if __name__ == "__main__":
    main()