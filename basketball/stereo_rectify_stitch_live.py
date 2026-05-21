#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stereo_rectify_stitch_live.py

在 stereo_calibrate_from_undistorted_images.py 已经得到的双目标定参数基础上，继续完成：

    1. 单目畸变矫正（可选，但实时读取原始摄像头时建议开启）
    2. stereoRectify 双目极线校正
    3. 生成并使用左右 remap 表
    4. 对左右校正图进行一次性背景特征匹配，估计右图到左图的 Homography
    5. 使用固定 Homography 做实时双目拼接
    6. 保存调试图、拼接图、Homography、stereoRectify 参数

为什么要分两级矫正？
------------------------------------------------------------
你的 stereo_calibrate_from_undistorted.npz 是基于“已经单目畸变矫正后的图片”计算的。
因此实时运行时如果直接从 USB 摄像头读取原始画面，需要先做一次单目畸变矫正，
把实时画面变成和当初双目标定输入图片相同的坐标系，然后再做 stereoRectify。

处理链路：

    原始左图  --> 左单目畸变矫正  --> 左双目极线校正  --> 左校正图
    原始右图  --> 右单目畸变矫正  --> 右双目极线校正  --> 右校正图
                                                        |
                                                        v
                                           SIFT/ORB + RANSAC 估计 H
                                                        |
                                                        v
                                             固定 H 实时拼接 panorama

重要说明：
------------------------------------------------------------
1. stereoRectify 解决的是“双目几何校正 / 极线对齐”，它不等于完整全景拼接。
2. 图像拼接还需要把右图变换到左图坐标系，这里用一次性的 Homography 完成。
3. Homography 建议在“空场背景图”上估计，不建议每帧全图 SIFT。
4. 人体、篮球、强反光、木地板重复纹理，不应该参与背景匹配。
5. 如果场地或机位变化，需要重新估计 Homography；双目 R/T 不一定需要重标定。

推荐首次使用方式：
------------------------------------------------------------
先用一对空场、已经畸变矫正后的左右图测试：

    python3 stereo_rectify_stitch_live.py \
        --mode image \
        --left-image  /home/elf/work/basketball/stereo_undistorted_calib/Left/align_000.jpg \
        --right-image /home/elf/work/basketball/stereo_undistorted_calib/Right/align_000.jpg \
        --stereo-file /home/elf/work/basketball/stereo_undistorted_calib/stereo_calibrate_from_undistorted.npz \
        --images-already-undistorted \
        --output-dir /home/elf/work/basketball/stereo_stitch_debug

实时运行，读取两个 USB 摄像头原始画面并先做单目畸变矫正：

    python3 stereo_rectify_stitch_live.py \
        --mode live \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
        --right-calib-file /home/elf/work/basketball/camera_calib.npz \
        --stereo-file /home/elf/work/basketball/stereo_undistorted_calib/stereo_calibrate_from_undistorted.npz \
        --output-dir /home/elf/work/basketball/stereo_stitch_debug \
        --display-scale 0.25

如果你已经让摄像头输出的就是“单目畸变矫正后”的图，可以加：

    --skip-mono-undistort

按键：
------------------------------------------------------------
    q / Esc : 退出
    s       : 保存当前 panorama、rectified left/right、side-by-side
    h       : 用当前帧重新估计 Homography
    r       : 切换显示 rectified side-by-side / panorama

依赖：
------------------------------------------------------------
    Python 3
    OpenCV
    NumPy
"""

import argparse
import os
import sys
import time
import select
import termios
import tty
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 1. 默认路径与参数
# ============================================================

# 这些默认值来自你现有脚本的工程目录习惯。
# 左相机：calib_usb2_camera.py 默认保存到 stereo_undistorted_calib/Left
# 右相机：calib_usb_camera.py  默认保存到 stereo_undistorted_calib/Right
DEFAULT_STEREO_FILE = "/home/elf/work/basketball/stereo_undistorted_calib/stereo_calibrate_from_undistorted.npz"
DEFAULT_LEFT_CALIB_FILE = "/home/elf/work/basketball/camera_usb2_calib.npz"
DEFAULT_RIGHT_CALIB_FILE = "/home/elf/work/basketball/camera_calib.npz"
DEFAULT_OUTPUT_DIR = "/home/elf/work/basketball/stereo_stitch_debug"

DEFAULT_LEFT_DEVICE = "/dev/video41"
DEFAULT_RIGHT_DEVICE = "/dev/video43"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30


# ============================================================
# 2. 小工具：FPS 与终端按键
# ============================================================

class FPSCounter:
    """简单 FPS 统计器。"""

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


class TerminalKeyReader:
    """
    非阻塞终端按键读取器。

    为什么需要这个？
        RK3588 + OpenCV imshow 时，经常会出现窗口拿不到键盘焦点，
        导致 cv2.waitKey() 读不到按键。
        这个类可以同时从终端读取 q/s/h/r 等按键。
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


# ============================================================
# 3. 摄像头打开
# ============================================================

def open_usb_camera(device: str, width: int, height: int, fps: int, use_mjpg: bool = True) -> cv2.VideoCapture:
    """
    使用 V4L2 打开 USB 摄像头。

    你的 RK3588 环境里，之前已经遇到 Python OpenCV 不支持 GStreamer 的问题，
    所以这里直接使用 cv2.CAP_V4L2。

    MJPG 的意义：
        1920x1080@30 如果使用未压缩 YUYV，USB2.0 带宽压力很大；
        MJPG 可以显著降低 USB 带宽压力。
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
    real_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((real_fourcc >> 8 * i) & 0xFF) for i in range(4))

    print(f"摄像头已打开: {device}")
    print(f"  size  : {real_w}x{real_h}")
    print(f"  fps   : {real_fps}")
    print(f"  fourcc: {fourcc_str}")

    if (real_w, real_h) != (width, height):
        print("[警告] 实际分辨率和请求分辨率不一致。")
        print(f"  request: {width}x{height}")
        print(f"  real   : {real_w}x{real_h}")

    return cap


# ============================================================
# 4. 单目畸变矫正器
# ============================================================

@dataclass
class MonoUndistorter:
    """
    单个相机的畸变矫正器。

    作用：
        把 USB 摄像头读到的“原始畸变图像”变成“已畸变矫正图像”。

    注意：
        你的 stereo_calibrate_from_undistorted.npz 是基于已畸变矫正图计算的，
        所以实时 raw 输入必须先经过这个步骤。
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    map1: np.ndarray
    map2: np.ndarray
    roi: Tuple[int, int, int, int]

    @staticmethod
    def from_file(calib_file: str, image_size: Tuple[int, int], alpha: float = 0.0, dist_scale: float = 0.8):
        """
        从单目标定 npz 文件创建畸变矫正器。

        dist_scale 与你原来的 calib_usb_camera.py / calib_usb2_camera.py 保持一致：
            默认 0.8。

        如果你当时生成 stereo_calibrate_from_undistorted.npz 用的是 dist_scale=1.0，
        这里实时运行也应该改成 --mono-dist-scale 1.0。
        两边必须一致，否则双目参数和实时画面坐标系对不上。
        """
        if not os.path.exists(calib_file):
            raise RuntimeError(f"找不到单目标定文件: {calib_file}")

        data = np.load(calib_file)
        K = data["camera_matrix"]
        D_raw = data["dist_coeffs"]
        saved_size = tuple(data["image_size"].astype(int))

        if saved_size != image_size:
            print("[警告] 单目标定分辨率和当前输入分辨率不一致。")
            print(f"  calib image_size: {saved_size}")
            print(f"  current size     : {image_size}")
            print("  建议使用相同分辨率采集、标定、运行。")

        # 复制畸变系数并按比例缩放径向畸变项。
        # OpenCV 常见顺序：[k1, k2, p1, p2, k3]
        # 只缩放 k1/k2/k3，不缩放切向畸变 p1/p2。
        D = D_raw.copy()
        flat = D.reshape(-1)
        if len(flat) >= 1:
            flat[0] *= dist_scale
        if len(flat) >= 2:
            flat[1] *= dist_scale
        if len(flat) >= 5:
            flat[4] *= dist_scale
        D = flat.reshape(D_raw.shape)

        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, image_size, alpha, image_size)

        map1, map2 = cv2.initUndistortRectifyMap(
            K,
            D,
            None,
            new_K,
            image_size,
            cv2.CV_16SC2
        )

        print(f"单目畸变矫正器已创建: {calib_file}")
        print(f"  dist_scale: {dist_scale}")
        print(f"  alpha     : {alpha}")
        print(f"  roi       : {roi}")

        return MonoUndistorter(K, D, image_size, map1, map2, tuple(int(x) for x in roi))

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """执行单目畸变矫正。"""
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


# ============================================================
# 5. 双目 stereoRectify + remap
# ============================================================

@dataclass
class StereoRectifier:
    """
    双目极线校正器。

    输入：
        左右“已单目畸变矫正”的图像。

    输出：
        左右“极线校正后”的图像。

    它做的事情：
        1. 从 stereo_calibrate_from_undistorted.npz 读取 K_left/K_right/R/T 等参数
        2. 调用 cv2.stereoRectify 得到 R1/R2/P1/P2/Q
        3. 调用 cv2.initUndistortRectifyMap 得到左右 remap 表
        4. 每帧使用 cv2.remap 快速校正
    """

    image_size: Tuple[int, int]
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray
    valid_roi_left: Tuple[int, int, int, int]
    valid_roi_right: Tuple[int, int, int, int]
    rms_stereo: Optional[float]
    baseline: float

    @staticmethod
    def from_stereo_file(stereo_file: str, rectify_alpha: float = 0.0, zero_disparity: bool = True):
        if not os.path.exists(stereo_file):
            raise RuntimeError(f"找不到双目标定文件: {stereo_file}")

        data = np.load(stereo_file)

        # 这些 key 是 stereo_calibrate_from_undistorted_images.py 保存出来的。
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

        # stereoRectify 的作用：
        #   把左右相机图像旋转到一个共同的“虚拟相机平面”上，
        #   使同一个空间点在左右图上的 y 坐标尽量一致。
        #
        # alpha:
        #   0   ：裁掉黑边，图像更干净，适合后续匹配和显示
        #   1   ：保留更多视野，但边缘会有黑边
        #   -1  ：OpenCV 自动选择
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
            newImageSize=image_size
        )

        # 因为 stereo npz 是从“已畸变矫正图片”算出来的，D_left/D_right 通常是 0。
        # 这里的 initUndistortRectifyMap 实际主要完成的是“极线校正旋转”。
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            K_left, D_left, R1, P1, image_size, cv2.CV_16SC2
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            K_right, D_right, R2, P2, image_size, cv2.CV_16SC2
        )

        print("双目 stereoRectify 已完成")
        print(f"  stereo_file  : {stereo_file}")
        print(f"  image_size   : {image_size}")
        print(f"  baseline     : {baseline:.3f} mm，单位来自你的 square_size")
        if rms_stereo is not None:
            print(f"  rms_stereo   : {rms_stereo:.4f} px")
            if rms_stereo > 2.0:
                print("  [提醒] rms_stereo > 2 px，建议先重点检查角点、左右图片同步和采集质量。")
        print(f"  roi_left     : {tuple(int(x) for x in roi_left)}")
        print(f"  roi_right    : {tuple(int(x) for x in roi_right)}")

        return StereoRectifier(
            image_size=image_size,
            R1=R1,
            R2=R2,
            P1=P1,
            P2=P2,
            Q=Q,
            map1_left=map1_left,
            map2_left=map2_left,
            map1_right=map1_right,
            map2_right=map2_right,
            valid_roi_left=tuple(int(x) for x in roi_left),
            valid_roi_right=tuple(int(x) for x in roi_right),
            rms_stereo=rms_stereo,
            baseline=baseline,
        )

    def rectify(self, left_undistorted: np.ndarray, right_undistorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对左右已单目畸变矫正图进行双目极线校正。"""
        left_rect = cv2.remap(left_undistorted, self.map1_left, self.map2_left, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_undistorted, self.map1_right, self.map2_right, interpolation=cv2.INTER_LINEAR)
        return left_rect, right_rect

    def save_rectify_params(self, path: str):
        """保存 stereoRectify 结果，方便后续程序直接加载。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            image_size=np.array(self.image_size),
            R1=self.R1,
            R2=self.R2,
            P1=self.P1,
            P2=self.P2,
            Q=self.Q,
            valid_roi_left=np.array(self.valid_roi_left),
            valid_roi_right=np.array(self.valid_roi_right),
            rms_stereo=np.array(-1.0 if self.rms_stereo is None else self.rms_stereo),
            baseline=np.array(self.baseline),
        )
        print(f"stereoRectify 参数已保存: {path}")


# ============================================================
# 6. 特征匹配 mask：尽量只匹配固定背景区域
# ============================================================

def parse_rects(rects_text: str) -> List[Tuple[int, int, int, int]]:
    """
    解析 ROI 字符串。

    格式：
        "x,y,w,h"
        "x,y,w,h;x,y,w,h"

    例子：
        --match-roi-left "0,0,1920,450;0,450,500,300;1400,450,520,300"

    用途：
        只在广告牌、篮架、墙体、看台这些固定背景区域提取特征，
        尽量避开木地板、白线、人和强反光。
    """
    rects_text = (rects_text or "").strip()
    if not rects_text:
        return []

    rects = []
    for part in rects_text.split(";"):
        vals = [int(v.strip()) for v in part.split(",")]
        if len(vals) != 4:
            raise ValueError(f"ROI 格式错误: {part}，应为 x,y,w,h")
        x, y, w, h = vals
        rects.append((x, y, w, h))
    return rects


def make_feature_mask(shape: Tuple[int, int], rects: List[Tuple[int, int, int, int]], ignore_bottom_ratio: float = 0.0) -> np.ndarray:
    """
    创建特征检测 mask。

    参数：
        shape:
            图像 shape[:2]，即 (height, width)。

        rects:
            允许提取特征的矩形区域列表。
            如果为空，默认全图可用。

        ignore_bottom_ratio:
            忽略图像底部多少比例。
            例如 0.35 表示底部 35% 不参与匹配，适合避开篮球场地板和人。
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if rects:
        for x, y, rw, rh in rects:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + rw)
            y2 = min(h, y + rh)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
    else:
        mask[:, :] = 255

    if ignore_bottom_ratio > 0:
        cut_y = int(h * (1.0 - ignore_bottom_ratio))
        mask[cut_y:, :] = 0

    return mask


# ============================================================
# 7. SIFT/ORB + RANSAC 估计右图到左图 Homography
# ============================================================

@dataclass
class HomographyResult:
    H_right_to_left: np.ndarray
    method_name: str
    total_matches: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    mean_reproj_error: float


def create_feature_detector(max_features: int = 5000):
    """
    创建特征检测器。

    优先使用 SIFT：
        对尺度、旋转、光照变化更稳，适合你的双目拼接初始化。

    如果当前 OpenCV 没有 SIFT，则回退 ORB：
        速度更快，但对视角/光照变化不如 SIFT。
    """
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
        method_name = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING
        method_name = "ORB"

    return detector, norm_type, method_name


def estimate_homography_by_features(
    left_img: np.ndarray,
    right_img: np.ndarray,
    mask_left: Optional[np.ndarray] = None,
    mask_right: Optional[np.ndarray] = None,
    max_features: int = 5000,
    ratio: float = 0.75,
    ransac_thresh: float = 4.0,
    min_good_matches: int = 30,
    min_inliers: int = 20,
    debug_match_path: Optional[str] = None,
) -> HomographyResult:
    """
    用特征点估计 H_right_to_left。

    H_right_to_left 的含义：
        输入右图上的点 p_right，变换到左图坐标系：
            p_left ~= H_right_to_left * p_right

    为什么是右图到左图？
        拼接时通常把左图作为参考底图，然后把右图 warp 到左图坐标系。
    """
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    detector, norm_type, method_name = create_feature_detector(max_features)

    kp_left, des_left = detector.detectAndCompute(gray_left, mask_left)
    kp_right, des_right = detector.detectAndCompute(gray_right, mask_right)

    if des_left is None or des_right is None or len(kp_left) < 8 or len(kp_right) < 8:
        raise RuntimeError(
            f"特征点太少，无法估计 Homography。"
            f" left_kp={len(kp_left)}, right_kp={len(kp_right)}。"
            f"建议换空场图，或调整 --match-roi-left/right。"
        )

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    # knnMatch：每个左图特征点在右图中找两个最近邻，后面用 Lowe ratio test 过滤误匹配。
    raw_matches = matcher.knnMatch(des_left, des_right, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_good_matches:
        raise RuntimeError(
            f"有效匹配太少: good={len(good)}, min_good_matches={min_good_matches}。"
            f"建议不要匹配木地板/白线，优先框选广告牌、墙体、篮架等固定背景。"
        )

    # 注意这里 queryIdx 是 left，trainIdx 是 right。
    # findHomography(src, dst) 需要 src=right，dst=left，才能得到 right_to_left。
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, ransac_thresh)

    if H is None or inlier_mask is None:
        raise RuntimeError("RANSAC 未能估计出 Homography。")

    inlier_mask_flat = inlier_mask.reshape(-1).astype(bool)
    inliers = int(np.count_nonzero(inlier_mask_flat))
    inlier_ratio = inliers / max(1, len(good))

    if inliers < min_inliers:
        raise RuntimeError(
            f"RANSAC 内点太少: inliers={inliers}, min_inliers={min_inliers}。"
            f"这通常说明两图重叠区域特征不稳定，或者左右图不是同一时刻/同一区域。"
        )

    # 计算内点重投影误差，越小越好。
    projected = cv2.perspectiveTransform(pts_right, H)
    errors = np.linalg.norm(projected - pts_left, axis=2).reshape(-1)
    mean_reproj_error = float(np.mean(errors[inlier_mask_flat])) if inliers > 0 else 9999.0

    if debug_match_path:
        os.makedirs(os.path.dirname(debug_match_path), exist_ok=True)
        draw_mask = inlier_mask.reshape(-1).tolist()
        debug = cv2.drawMatches(
            left_img,
            kp_left,
            right_img,
            kp_right,
            good,
            None,
            matchesMask=draw_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(debug_match_path, debug)

    return HomographyResult(
        H_right_to_left=H,
        method_name=method_name,
        total_matches=len(raw_matches),
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(inlier_ratio),
        mean_reproj_error=mean_reproj_error,
    )


# ============================================================
# 8. 计算拼接画布 + 图像融合
# ============================================================

@dataclass
class StitchLayout:
    """
    拼接布局。

    H_right_to_canvas:
        右图到最终 canvas 的 Homography。

    offset_x / offset_y:
        左图被放到 canvas 里的偏移量。
        因为右图 warp 后可能出现负坐标，所以需要整体平移。

    canvas_size:
        cv2.warpPerspective 使用的尺寸，格式为 (width, height)。
    """

    H_right_to_canvas: np.ndarray
    offset_x: int
    offset_y: int
    canvas_size: Tuple[int, int]


def compute_stitch_layout(left_shape: Tuple[int, int, int], right_shape: Tuple[int, int, int], H_right_to_left: np.ndarray) -> StitchLayout:
    """
    根据 H 计算最终 panorama 画布大小。

    思路：
        1. 左图四个角点在左图坐标系中不变
        2. 右图四个角点通过 H 映射到左图坐标系
        3. 计算所有角点的 min/max，得到能容纳两张图的画布
        4. 如果 min_x/min_y 为负，则整体平移到正坐标
    """
    h_left, w_left = left_shape[:2]
    h_right, w_right = right_shape[:2]

    corners_left = np.float32([
        [0, 0],
        [w_left, 0],
        [w_left, h_left],
        [0, h_left],
    ]).reshape(-1, 1, 2)

    corners_right = np.float32([
        [0, 0],
        [w_right, 0],
        [w_right, h_right],
        [0, h_right],
    ]).reshape(-1, 1, 2)

    warped_right_corners = cv2.perspectiveTransform(corners_right, H_right_to_left)
    all_corners = np.vstack([corners_left, warped_right_corners]).reshape(-1, 2)

    min_x = float(np.floor(np.min(all_corners[:, 0])))
    min_y = float(np.floor(np.min(all_corners[:, 1])))
    max_x = float(np.ceil(np.max(all_corners[:, 0])))
    max_y = float(np.ceil(np.max(all_corners[:, 1])))

    offset_x = int(-min_x) if min_x < 0 else 0
    offset_y = int(-min_y) if min_y < 0 else 0

    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)

    T_offset = np.array([
        [1.0, 0.0, offset_x],
        [0.0, 1.0, offset_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H_right_to_canvas = T_offset @ H_right_to_left

    return StitchLayout(
        H_right_to_canvas=H_right_to_canvas,
        offset_x=offset_x,
        offset_y=offset_y,
        canvas_size=(canvas_w, canvas_h),
    )


def image_valid_mask(img: np.ndarray) -> np.ndarray:
    """
    生成图像有效区域 mask。

    黑边通常接近 [0,0,0]，这里用灰度 > 0 判断有效区域。
    如果你的场景里有非常黑的区域，这个判断可能会误伤；篮球馆通常问题不大。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 0).astype(np.uint8) * 255


def warp_and_blend(
    left_img: np.ndarray,
    right_img: np.ndarray,
    H_right_to_left: np.ndarray,
    blend_mode: str = "feather",
    seam_x: Optional[int] = None,
) -> Tuple[np.ndarray, StitchLayout]:
    """
    把右图 warp 到左图坐标系，并融合成 panorama。

    blend_mode:
        feather:
            重叠区域用距离权重渐变融合，视觉更自然，但人经过接缝可能出现半透明重影。

        left-priority:
            重叠区域优先使用左图，适合接缝附近主要关注左相机检测结果。

        right-priority:
            重叠区域优先使用右图。

        cut:
            使用一条竖直接缝 seam_x，左边用左图，右边用右图。
            这种方式更适合避免运动人体在接缝处被双重融合。
    """
    layout = compute_stitch_layout(left_img.shape, right_img.shape, H_right_to_left)
    canvas_w, canvas_h = layout.canvas_size

    # 右图直接透视变换到 canvas。
    right_warp = cv2.warpPerspective(
        right_img,
        layout.H_right_to_canvas,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 左图不做透视变换，只是放到 canvas 的 offset 位置。
    left_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    h_left, w_left = left_img.shape[:2]
    x0, y0 = layout.offset_x, layout.offset_y
    left_canvas[y0:y0 + h_left, x0:x0 + w_left] = left_img

    mask_left = image_valid_mask(left_canvas)
    mask_right = image_valid_mask(right_warp)

    only_left = (mask_left > 0) & (mask_right == 0)
    only_right = (mask_right > 0) & (mask_left == 0)
    overlap = (mask_left > 0) & (mask_right > 0)

    panorama = np.zeros_like(left_canvas)
    panorama[only_left] = left_canvas[only_left]
    panorama[only_right] = right_warp[only_right]

    if np.any(overlap):
        if blend_mode == "left-priority":
            panorama[overlap] = left_canvas[overlap]

        elif blend_mode == "right-priority":
            panorama[overlap] = right_warp[overlap]

        elif blend_mode == "cut":
            # seam_x 是 canvas 坐标。
            # 如果没有指定，默认取重叠区域 x 坐标中点。
            ys, xs = np.where(overlap)
            if seam_x is None:
                seam_x_use = int((np.min(xs) + np.max(xs)) / 2)
            else:
                seam_x_use = int(seam_x)

            left_side = overlap & (np.indices(mask_left.shape)[1] <= seam_x_use)
            right_side = overlap & (np.indices(mask_left.shape)[1] > seam_x_use)
            panorama[left_side] = left_canvas[left_side]
            panorama[right_side] = right_warp[right_side]

        else:
            # feather 融合：
            # 距离各自 mask 边缘越远，权重越大。
            # 注意 distanceTransform 输入需要 8-bit 单通道 0/255。
            dist_left = cv2.distanceTransform(mask_left, cv2.DIST_L2, 3).astype(np.float32)
            dist_right = cv2.distanceTransform(mask_right, cv2.DIST_L2, 3).astype(np.float32)

            denom = dist_left + dist_right + 1e-6
            weight_left = dist_left / denom
            weight_right = dist_right / denom

            # 只在 overlap 中使用渐变权重。
            wl = weight_left[..., None]
            wr = weight_right[..., None]
            blended = left_canvas.astype(np.float32) * wl + right_warp.astype(np.float32) * wr
            panorama[overlap] = np.clip(blended, 0, 255).astype(np.uint8)[overlap]

    return panorama, layout


# ============================================================
# 9. 调试显示辅助
# ============================================================

def draw_epipolar_lines(left_rect: np.ndarray, right_rect: np.ndarray, step: int = 80) -> np.ndarray:
    """
    把左右校正图左右拼起来，并画水平线。

    目的：
        检查 stereoRectify 是否把极线对齐。
        理想情况下，同一个物体在左右图里的对应点应该落在同一条水平线上。
    """
    h = min(left_rect.shape[0], right_rect.shape[0])
    w_left = left_rect.shape[1]

    left = left_rect[:h].copy()
    right = right_rect[:h].copy()
    combined = np.hstack([left, right])

    for y in range(0, h, step):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 255), 1)

    cv2.line(combined, (w_left, 0), (w_left, h), (255, 255, 255), 2)
    return combined


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def save_debug_images(output_dir: str, prefix: str, left_rect: np.ndarray, right_rect: np.ndarray, side_by_side: np.ndarray, panorama: Optional[np.ndarray]):
    """保存当前调试图。"""
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_left_rect.jpg"), left_rect)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_right_rect.jpg"), right_rect)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_side_by_side.jpg"), side_by_side)
    if panorama is not None:
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_panorama.jpg"), panorama)


# ============================================================
# 10. Homography 保存/读取
# ============================================================

def save_homography(path: str, H: np.ndarray, result: HomographyResult, layout: Optional[StitchLayout] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {
        "H_right_to_left": H,
        "method_name": np.array(result.method_name),
        "total_matches": np.array(result.total_matches),
        "good_matches": np.array(result.good_matches),
        "inliers": np.array(result.inliers),
        "inlier_ratio": np.array(result.inlier_ratio),
        "mean_reproj_error": np.array(result.mean_reproj_error),
    }
    if layout is not None:
        save_dict.update({
            "H_right_to_canvas": layout.H_right_to_canvas,
            "offset_x": np.array(layout.offset_x),
            "offset_y": np.array(layout.offset_y),
            "canvas_size": np.array(layout.canvas_size),
        })
    np.savez(path, **save_dict)
    print(f"Homography 已保存: {path}")


def load_homography(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise RuntimeError(f"找不到 Homography 文件: {path}")
    data = np.load(path, allow_pickle=True)
    return data["H_right_to_left"]


# ============================================================
# 11. 单帧处理链路
# ============================================================

def prepare_frames(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    left_undistorter: Optional[MonoUndistorter],
    right_undistorter: Optional[MonoUndistorter],
    stereo_rectifier: StereoRectifier,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    单帧处理：
        原始帧 -> 单目畸变矫正 -> 双目极线校正
    """
    if raw_left is None or raw_right is None:
        raise RuntimeError("输入帧为空")

    # 如果 left_undistorter/right_undistorter 为 None，说明输入已经是已畸变矫正图。
    if left_undistorter is not None:
        left_undistorted = left_undistorter.undistort(raw_left)
    else:
        left_undistorted = raw_left

    if right_undistorter is not None:
        right_undistorted = right_undistorter.undistort(raw_right)
    else:
        right_undistorted = raw_right

    # 双目极线校正。
    left_rect, right_rect = stereo_rectifier.rectify(left_undistorted, right_undistorted)

    return left_rect, right_rect


# ============================================================
# 12. image 模式：处理一对图片
# ============================================================

def run_image_mode(args):
    if not args.left_image or not args.right_image:
        raise RuntimeError("image 模式必须指定 --left-image 和 --right-image")

    os.makedirs(args.output_dir, exist_ok=True)

    stereo_rectifier = StereoRectifier.from_stereo_file(
        args.stereo_file,
        rectify_alpha=args.rectify_alpha,
        zero_disparity=not args.no_zero_disparity,
    )
    stereo_rectifier.save_rectify_params(os.path.join(args.output_dir, "stereo_rectify_params.npz"))

    left = cv2.imread(args.left_image)
    right = cv2.imread(args.right_image)

    if left is None:
        raise RuntimeError(f"左图读取失败: {args.left_image}")
    if right is None:
        raise RuntimeError(f"右图读取失败: {args.right_image}")

    image_size = stereo_rectifier.image_size
    if (left.shape[1], left.shape[0]) != image_size:
        print("[警告] 左图尺寸和 stereo npz 中 image_size 不一致，自动 resize。")
        left = cv2.resize(left, image_size)
    if (right.shape[1], right.shape[0]) != image_size:
        print("[警告] 右图尺寸和 stereo npz 中 image_size 不一致，自动 resize。")
        right = cv2.resize(right, image_size)

    # image 模式通常用于已经保存好的“畸变矫正图”。
    # 如果输入是 raw 原图，则去掉 --images-already-undistorted，让脚本读取单目标定文件先矫正。
    if args.images_already_undistorted:
        left_undistorter = None
        right_undistorter = None
    else:
        left_undistorter = MonoUndistorter.from_file(args.left_calib_file, image_size, args.mono_alpha, args.mono_dist_scale)
        right_undistorter = MonoUndistorter.from_file(args.right_calib_file, image_size, args.mono_alpha, args.mono_dist_scale)

    left_rect, right_rect = prepare_frames(left, right, left_undistorter, right_undistorter, stereo_rectifier)
    side_by_side = draw_epipolar_lines(left_rect, right_rect, step=args.epiline_step)

    # 准备特征匹配区域。
    roi_left = parse_rects(args.match_roi_left)
    roi_right = parse_rects(args.match_roi_right)
    mask_left = make_feature_mask(left_rect.shape, roi_left, args.ignore_bottom_ratio)
    mask_right = make_feature_mask(right_rect.shape, roi_right, args.ignore_bottom_ratio)

    # 如果指定了已有 Homography，就直接加载；否则从当前图片估计。
    if args.homography_file and os.path.exists(args.homography_file) and not args.force_reestimate_h:
        H = load_homography(args.homography_file)
        print(f"已加载 Homography: {args.homography_file}")
        fake_result = HomographyResult(H, "loaded", 0, 0, 0, 0.0, 0.0)
    else:
        match_debug = os.path.join(args.output_dir, "debug_feature_matches.jpg")
        result = estimate_homography_by_features(
            left_rect,
            right_rect,
            mask_left=mask_left,
            mask_right=mask_right,
            max_features=args.max_features,
            ratio=args.ratio,
            ransac_thresh=args.ransac_thresh,
            min_good_matches=args.min_good_matches,
            min_inliers=args.min_inliers,
            debug_match_path=match_debug,
        )
        H = result.H_right_to_left
        fake_result = result
        print("Homography 估计完成")
        print(f"  method            : {result.method_name}")
        print(f"  total_matches     : {result.total_matches}")
        print(f"  good_matches      : {result.good_matches}")
        print(f"  inliers           : {result.inliers}")
        print(f"  inlier_ratio      : {result.inlier_ratio:.3f}")
        print(f"  mean_reproj_error : {result.mean_reproj_error:.3f} px")

    panorama, layout = warp_and_blend(left_rect, right_rect, H, blend_mode=args.blend_mode, seam_x=args.seam_x)

    if args.homography_file:
        save_homography(args.homography_file, H, fake_result, layout)
    else:
        save_homography(os.path.join(args.output_dir, "homography_right_to_left.npz"), H, fake_result, layout)

    save_debug_images(args.output_dir, "image", left_rect, right_rect, side_by_side, panorama)

    print("\nimage 模式处理完成，输出目录:")
    print(args.output_dir)
    print("主要文件：")
    print("  image_left_rect.jpg")
    print("  image_right_rect.jpg")
    print("  image_side_by_side.jpg")
    print("  image_panorama.jpg")
    print("  debug_feature_matches.jpg")
    print("  homography_right_to_left.npz")

    if not args.headless:
        cv2.imshow("rectified side-by-side", resize_for_display(side_by_side, args.display_scale))
        cv2.imshow("panorama", resize_for_display(panorama, args.display_scale))
        print("按任意键关闭窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# 13. live 模式：实时双目拼接
# ============================================================

def run_live_mode(args):
    os.makedirs(args.output_dir, exist_ok=True)

    stereo_rectifier = StereoRectifier.from_stereo_file(
        args.stereo_file,
        rectify_alpha=args.rectify_alpha,
        zero_disparity=not args.no_zero_disparity,
    )
    stereo_rectifier.save_rectify_params(os.path.join(args.output_dir, "stereo_rectify_params.npz"))

    image_size = stereo_rectifier.image_size
    if (args.width, args.height) != image_size:
        print("[警告] 当前采集分辨率和 stereo npz 中 image_size 不一致。")
        print(f"  capture     : {(args.width, args.height)}")
        print(f"  stereo npz  : {image_size}")
        print("  建议用相同分辨率运行，否则会自动 resize，精度会变差。")

    # 实时 raw 输入默认需要先做单目畸变矫正。
    # 如果你已经在前级程序里输出了已畸变矫正图，则加 --skip-mono-undistort。
    if args.skip_mono_undistort:
        left_undistorter = None
        right_undistorter = None
        print("已跳过单目畸变矫正：假设摄像头输入已经是畸变矫正后的图。")
    else:
        left_undistorter = MonoUndistorter.from_file(args.left_calib_file, image_size, args.mono_alpha, args.mono_dist_scale)
        right_undistorter = MonoUndistorter.from_file(args.right_calib_file, image_size, args.mono_alpha, args.mono_dist_scale)

    cap_left = open_usb_camera(args.left_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)
    cap_right = open_usb_camera(args.right_device, args.width, args.height, args.fps, use_mjpg=not args.no_mjpg)

    # 加载已有 H 或等待第一帧估计 H。
    H = None
    if args.homography_file and os.path.exists(args.homography_file) and not args.force_reestimate_h:
        H = load_homography(args.homography_file)
        print(f"已加载 Homography: {args.homography_file}")

    roi_left = parse_rects(args.match_roi_left)
    roi_right = parse_rects(args.match_roi_right)

    frame_idx = 0
    save_idx = 0
    fps_counter = FPSCounter()
    show_mode = "panorama"  # panorama 或 rectified

    print("\n开始实时双目校正 + 拼接")
    print("按键：q/Esc 退出，s 保存，h 重新估计 Homography，r 切换显示。")

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

            # 如果实际采集尺寸不是 stereo image_size，resize 到 stereo 标定尺寸。
            # 更好的做法是让采集分辨率和标定分辨率完全一致。
            if (raw_left.shape[1], raw_left.shape[0]) != image_size:
                raw_left = cv2.resize(raw_left, image_size)
            if (raw_right.shape[1], raw_right.shape[0]) != image_size:
                raw_right = cv2.resize(raw_right, image_size)

            left_rect, right_rect = prepare_frames(raw_left, raw_right, left_undistorter, right_undistorter, stereo_rectifier)
            side_by_side = draw_epipolar_lines(left_rect, right_rect, step=args.epiline_step)

            # 第一帧没有 H 时，估计一次。
            # 后续默认不每帧估计，除非设置 --update-h-every。
            need_estimate = H is None
            if args.update_h_every > 0 and frame_idx > 0 and frame_idx % args.update_h_every == 0:
                need_estimate = True

            if need_estimate:
                try:
                    mask_left = make_feature_mask(left_rect.shape, roi_left, args.ignore_bottom_ratio)
                    mask_right = make_feature_mask(right_rect.shape, roi_right, args.ignore_bottom_ratio)
                    match_debug = os.path.join(args.output_dir, f"debug_feature_matches_{frame_idx:06d}.jpg")

                    result = estimate_homography_by_features(
                        left_rect,
                        right_rect,
                        mask_left=mask_left,
                        mask_right=mask_right,
                        max_features=args.max_features,
                        ratio=args.ratio,
                        ransac_thresh=args.ransac_thresh,
                        min_good_matches=args.min_good_matches,
                        min_inliers=args.min_inliers,
                        debug_match_path=match_debug,
                    )
                    H = result.H_right_to_left

                    print("Homography 估计/更新完成")
                    print(f"  frame             : {frame_idx}")
                    print(f"  method            : {result.method_name}")
                    print(f"  good/inliers      : {result.good_matches}/{result.inliers}")
                    print(f"  inlier_ratio      : {result.inlier_ratio:.3f}")
                    print(f"  mean_reproj_error : {result.mean_reproj_error:.3f} px")

                    # 用当前 H 保存一份，下一次进场可以直接加载，避免启动时每次重新匹配。
                    if args.homography_file:
                        panorama_tmp, layout_tmp = warp_and_blend(left_rect, right_rect, H, blend_mode=args.blend_mode, seam_x=args.seam_x)
                        save_homography(args.homography_file, H, result, layout_tmp)

                except Exception as e:
                    print(f"[警告] Homography 估计失败: {e}")
                    print("当前先显示 rectified side-by-side。")

            panorama = None
            if H is not None:
                try:
                    panorama, _ = warp_and_blend(left_rect, right_rect, H, blend_mode=args.blend_mode, seam_x=args.seam_x)
                except Exception as e:
                    print(f"[警告] 拼接失败: {e}")
                    panorama = None

            fps = fps_counter.update()

            # 显示。
            if not args.headless:
                if show_mode == "panorama" and panorama is not None:
                    display = panorama.copy()
                    cv2.putText(display, f"Panorama | FPS:{fps:.1f} | q quit | s save | h re-H | r view", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("stereo panorama", resize_for_display(display, args.display_scale))
                else:
                    display = side_by_side.copy()
                    cv2.putText(display, f"Rectified | FPS:{fps:.1f} | q quit | s save | h re-H | r view", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("rectified side-by-side", resize_for_display(display, args.display_scale))

            # 读取按键。
            cv_key = -1
            if not args.headless:
                cv_key = cv2.waitKey(1)
            term_key = key_reader.read_key()

            key_char = None
            if cv_key != -1:
                key_char = chr(cv_key & 0xFF)
            if term_key is not None:
                key_char = term_key

            if key_char is not None:
                if key_char in ("q", "Q") or cv_key == 27:
                    print("退出实时拼接")
                    break

                elif key_char in ("s", "S"):
                    prefix = f"live_{save_idx:06d}"
                    save_debug_images(args.output_dir, prefix, left_rect, right_rect, side_by_side, panorama)
                    print(f"[SAVE] 已保存 {prefix}_*.jpg 到 {args.output_dir}")
                    save_idx += 1

                elif key_char in ("h", "H"):
                    print("收到 h：下一帧重新估计 Homography")
                    H = None

                elif key_char in ("r", "R"):
                    show_mode = "rectified" if show_mode == "panorama" else "panorama"
                    print(f"show_mode = {show_mode}")

            frame_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


# ============================================================
# 14. 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stereo rectify and panorama stitching based on stereo_calibrate_from_undistorted.npz"
    )

    parser.add_argument("--mode", choices=["image", "live"], required=True, help="image=处理一对图片；live=实时双摄像头")

    # 标定文件。
    parser.add_argument("--stereo-file", default=DEFAULT_STEREO_FILE, help="stereo_calibrate_from_undistorted.npz 路径")
    parser.add_argument("--left-calib-file", default=DEFAULT_LEFT_CALIB_FILE, help="左相机单目标定 npz，例如 camera_usb2_calib.npz")
    parser.add_argument("--right-calib-file", default=DEFAULT_RIGHT_CALIB_FILE, help="右相机单目标定 npz，例如 camera_calib.npz")

    # image 模式输入。
    parser.add_argument("--left-image", default="", help="image 模式左图路径")
    parser.add_argument("--right-image", default="", help="image 模式右图路径")
    parser.add_argument("--images-already-undistorted", action="store_true", help="image 输入是否已经完成单目畸变矫正")

    # live 模式摄像头参数。
    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备节点")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备节点")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="采集宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="采集高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="禁用 MJPG 请求")
    parser.add_argument("--skip-mono-undistort", action="store_true", help="实时输入已经是畸变矫正图时使用")

    # 单目畸变矫正参数。
    parser.add_argument("--mono-alpha", type=float, default=0.0, help="单目 getOptimalNewCameraMatrix alpha，建议和生成双目标定图时一致")
    parser.add_argument("--mono-dist-scale", type=float, default=0.8, help="单目畸变强度缩放，建议和生成双目标定图时一致")

    # stereoRectify 参数。
    parser.add_argument("--rectify-alpha", type=float, default=0.0, help="stereoRectify alpha：0裁黑边，1保留视野，-1自动")
    parser.add_argument("--no-zero-disparity", action="store_true", help="不使用 CALIB_ZERO_DISPARITY")
    parser.add_argument("--epiline-step", type=int, default=80, help="side-by-side 调试图水平线间隔")

    # Homography / 特征匹配参数。
    parser.add_argument("--homography-file", default="", help="保存/加载右图到左图 Homography 的 npz 文件")
    parser.add_argument("--force-reestimate-h", action="store_true", help="即使 homography 文件存在，也强制重新估计")
    parser.add_argument("--update-h-every", type=int, default=0, help="live 模式每隔 N 帧重新估计 H。0 表示不自动更新")
    parser.add_argument("--max-features", type=int, default=5000, help="SIFT/ORB 最大特征数量")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test 阈值")
    parser.add_argument("--ransac-thresh", type=float, default=4.0, help="findHomography RANSAC 阈值，单位像素")
    parser.add_argument("--min-good-matches", type=int, default=30, help="估计 H 所需最少 good matches")
    parser.add_argument("--min-inliers", type=int, default=20, help="估计 H 所需最少 RANSAC 内点")
    parser.add_argument("--match-roi-left", default="", help="左图特征匹配 ROI，格式 x,y,w,h;x,y,w,h")
    parser.add_argument("--match-roi-right", default="", help="右图特征匹配 ROI，格式 x,y,w,h;x,y,w,h")
    parser.add_argument("--ignore-bottom-ratio", type=float, default=0.0, help="忽略底部比例，如 0.35 可避开地板和人")

    # 融合参数。
    parser.add_argument("--blend-mode", choices=["feather", "left-priority", "right-priority", "cut"], default="feather", help="拼接重叠区融合方式")
    parser.add_argument("--seam-x", type=int, default=None, help="blend-mode=cut 时的 canvas 接缝 x 坐标，不填则取重叠区中线")

    # 输出与显示。
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="调试图输出目录")
    parser.add_argument("--display-scale", type=float, default=0.3, help="显示缩放比例")
    parser.add_argument("--headless", action="store_true", help="无显示窗口模式")

    return parser.parse_args()


# ============================================================
# 15. 主函数
# ============================================================

def main():
    args = parse_args()

    print("\n================ 当前参数 ================")
    print(f"mode              : {args.mode}")
    print(f"stereo_file       : {args.stereo_file}")
    print(f"left_calib_file   : {args.left_calib_file}")
    print(f"right_calib_file  : {args.right_calib_file}")
    print(f"output_dir        : {args.output_dir}")
    print(f"blend_mode        : {args.blend_mode}")
    print(f"mono_dist_scale   : {args.mono_dist_scale}")
    print(f"rectify_alpha     : {args.rectify_alpha}")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n[提示] 未检测到 DISPLAY，OpenCV 窗口可能无法显示。")
        print("可以先执行：export DISPLAY=:0")

    if args.mode == "image":
        run_image_mode(args)
    elif args.mode == "live":
        run_live_mode(args)


if __name__ == "__main__":
    main()
