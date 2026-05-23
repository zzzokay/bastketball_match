#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对“已经畸变矫正后的左右棋盘格图片”执行 stereoCalibrate。

当前脚本目标：
    1. 读取已经畸变矫正后的 left/right 棋盘格图片
    2. 检测左右图中的棋盘格角点
    3. 使用 cv2.calibrateCamera 分别估计左右图当前等效内参
    4. 使用 cv2.stereoCalibrate 计算双目外参 R、T、E、F
    5. 保存 stereoCalibrate 参数到 npz
    6. 保存角点检测调试图，方便你检查角点是否正确

注意：
    这个脚本暂时不做 stereoRectify。
    这个脚本暂时不生成 remap。
    这个脚本暂时不做最终拼接。

为什么这里还要 calibrateCamera？
    因为 stereoCalibrate 需要左右相机的内参 K 和畸变 D。
    你输入的是“已畸变矫正后的图片”，理论上畸变可以近似看成 0。
    但仍然需要一个当前矫正图像坐标系下的等效相机内参 K。
    因此这里先对已矫正图片做单目标定，得到 K_left、K_right，
    然后 stereoCalibrate 固定内参求 R、T。

适用场景：
    - 你已经用单目畸变矫正脚本保存了一批左右“矫正后”的棋盘格图片
    - 你现在只想观察 stereoCalibrate 的 R/T/E/F 结果
"""

import os
import glob
import argparse
import sys

import cv2
import numpy as np


# ============================================================
# 1. 默认配置
# ============================================================

DEFAULT_LEFT_DIR = "/home/elf/work/basketball/stereo_undistorted_calib/Left"
DEFAULT_RIGHT_DIR = "/home/elf/work/basketball/stereo_undistorted_calib/Right"

DEFAULT_OUTPUT = "/home/elf/work/basketball/stereo_undistorted_calib/stereo_calibrate_from_undistorted.npz"
DEFAULT_DEBUG_DIR = "/home/elf/work/basketball/stereo_undistorted_calib/debug_corners"

# 你的棋盘格是 12 x 9 个方格，所以 OpenCV 使用 11 x 8 个内角点
DEFAULT_BOARD_COLS = 11
DEFAULT_BOARD_ROWS = 8

# 你的棋盘格方格边长，单位 mm
DEFAULT_SQUARE_SIZE = 22.0


# ============================================================
# 2. 创建棋盘格世界坐标
# ============================================================

def create_object_points(board_cols, board_rows, square_size):
    """
    创建棋盘格角点在真实世界中的 3D 坐标。

    OpenCV 标定需要两类点：
        1. object_points:
            棋盘格角点在真实世界中的坐标，单位由 square_size 决定。
        2. image_points:
            棋盘格角点在图像中的像素坐标。

    对于平面棋盘格，假设棋盘格位于 Z=0 平面。

    例如：
        board_cols = 11
        board_rows = 8
        square_size = 22.0

    那么第一行角点坐标大致是：
        (0, 0, 0)
        (22, 0, 0)
        (44, 0, 0)
        ...
    """

    objp = np.zeros((board_rows * board_cols, 3), np.float32)

    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)

    objp[:, :2] = grid * square_size

    return objp


# ============================================================
# 3. 棋盘格角点检测
# ============================================================

def find_chessboard_corners(frame, board_cols, board_rows):
    """
    在图像中检测棋盘格内角点。

    参数：
        frame:
            BGR 图像。

        board_cols, board_rows:
            棋盘格内角点数量。
            注意不是方格数量。

    返回：
        found:
            是否检测成功。

        corners:
            检测到的角点，shape 通常是 (N, 1, 2)。

        gray:
            灰度图，方便调试或后续处理。

    检测策略：
        1. 优先使用 findChessboardCornersSB
           OpenCV 4.x 下通常更稳定。
        2. 如果失败，回退到传统 findChessboardCorners。
        3. 成功后用 cornerSubPix 做亚像素优化。
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 提升对比度，增加棋盘格检测稳定性
    gray_eq = cv2.equalizeHist(gray)

    pattern_size = (board_cols, board_rows)

    found = False
    corners = None

    # --------------------------------------------------------
    # 3.1 优先使用新版 SB 棋盘格检测
    # --------------------------------------------------------

    if hasattr(cv2, "findChessboardCornersSB"):
        flags_sb = (
            cv2.CALIB_CB_NORMALIZE_IMAGE |
            cv2.CALIB_CB_EXHAUSTIVE |
            cv2.CALIB_CB_ACCURACY
        )

        found, corners = cv2.findChessboardCornersSB(
            gray_eq,
            pattern_size,
            flags_sb
        )

    # --------------------------------------------------------
    # 3.2 如果 SB 失败，回退到传统方法
    # --------------------------------------------------------

    if not found:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        found, corners = cv2.findChessboardCorners(
            gray_eq,
            pattern_size,
            flags
        )

    if not found or corners is None:
        return False, None, gray

    # --------------------------------------------------------
    # 3.3 亚像素优化
    #
    # 注意：
    #     SB 返回的角点本身已经比较精细；
    #     传统方法一定建议 cornerSubPix。
    #     这里统一再 refine 一次。
    # --------------------------------------------------------

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    try:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )
    except cv2.error:
        # 某些 OpenCV 版本下 SB 返回的角点格式可能不适合再 refine。
        # 这种情况直接使用原 corners。
        pass

    return True, corners, gray


# ============================================================
# 4. 收集左右棋盘格角点
# ============================================================

def collect_stereo_chessboard_points(
    left_dir,
    right_dir,
    board_cols,
    board_rows,
    square_size,
    debug_dir
):
    """
    从左右图片目录中读取成对棋盘格图片，并检测角点。

    返回：
        object_points_all:
            每组图片对应的棋盘格 3D 点。

        image_points_left_all:
            左图检测到的 2D 角点。

        image_points_right_all:
            右图检测到的 2D 角点。

        image_size:
            图像尺寸，格式为 (width, height)。

        valid_pairs:
            有效图片对数量。
    """

    os.makedirs(debug_dir, exist_ok=True)

    # 支持 jpg/jpeg/png
    left_paths = []
    right_paths = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        left_paths.extend(glob.glob(os.path.join(left_dir, ext)))
        right_paths.extend(glob.glob(os.path.join(right_dir, ext)))

    left_paths = sorted(left_paths)
    right_paths = sorted(right_paths)

    if len(left_paths) == 0:
        raise RuntimeError(f"左图目录没有图片: {left_dir}")

    if len(right_paths) == 0:
        raise RuntimeError(f"右图目录没有图片: {right_dir}")

    if len(left_paths) != len(right_paths):
        raise RuntimeError(
            f"左右图片数量不一致: left={len(left_paths)}, right={len(right_paths)}"
        )

    print("左图数量:", len(left_paths))
    print("右图数量:", len(right_paths))

    objp = create_object_points(
        board_cols=board_cols,
        board_rows=board_rows,
        square_size=square_size
    )

    object_points_all = []
    image_points_left_all = []
    image_points_right_all = []

    image_size = None
    valid_pairs = 0

    for idx, (left_path, right_path) in enumerate(zip(left_paths, right_paths)):
        print("\n--------------------------------------------")
        print(f"处理第 {idx} 组")
        print("left :", left_path)
        print("right:", right_path)

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None:
            print("[跳过] 左图读取失败")
            continue

        if img_right is None:
            print("[跳过] 右图读取失败")
            continue

        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        current_size_left = (w_left, h_left)
        current_size_right = (w_right, h_right)

        if image_size is None:
            image_size = current_size_left
            print("图像尺寸:", image_size)

        if current_size_left != image_size:
            print("[跳过] 左图尺寸与第一张不一致:", current_size_left)
            continue

        if current_size_right != image_size:
            print("[跳过] 右图尺寸与第一张不一致:", current_size_right)
            continue

        # ----------------------------------------------------
        # 4.1 检测左右棋盘格角点
        # ----------------------------------------------------

        found_left, corners_left, _ = find_chessboard_corners(
            img_left,
            board_cols,
            board_rows
        )

        found_right, corners_right, _ = find_chessboard_corners(
            img_right,
            board_cols,
            board_rows
        )

        if found_left and found_right:
            print("[OK] 左右角点检测成功")

            object_points_all.append(objp.copy())
            image_points_left_all.append(corners_left)
            image_points_right_all.append(corners_right)

            valid_pairs += 1

            # ------------------------------------------------
            # 保存角点检测可视化图，方便你人工检查
            # ------------------------------------------------

            debug_left = img_left.copy()
            debug_right = img_right.copy()

            cv2.drawChessboardCorners(
                debug_left,
                (board_cols, board_rows),
                corners_left,
                True
            )

            cv2.drawChessboardCorners(
                debug_right,
                (board_cols, board_rows),
                corners_right,
                True
            )

            cv2.imwrite(
                os.path.join(debug_dir, f"left_corners_{idx:03d}.jpg"),
                debug_left
            )

            cv2.imwrite(
                os.path.join(debug_dir, f"right_corners_{idx:03d}.jpg"),
                debug_right
            )

        else:
            print("[FAIL] 角点检测失败")
            print("left found :", found_left)
            print("right found:", found_right)

            # 保存失败图，方便排查
            cv2.imwrite(
                os.path.join(debug_dir, f"left_fail_{idx:03d}.jpg"),
                img_left
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"right_fail_{idx:03d}.jpg"),
                img_right
            )

    print("\n有效左右图片对数量:", valid_pairs)

    return (
        object_points_all,
        image_points_left_all,
        image_points_right_all,
        image_size,
        valid_pairs
    )


# ============================================================
# 5. 单目标定
# ============================================================

def calibrate_single_camera(object_points_all, image_points_all, image_size, name):
    """
    对单个相机进行 calibrateCamera。

    这里输入的是已经畸变矫正后的图片。
    因此求出来的 dist_coeffs 理论上应该接近 0。

    为什么仍然执行 calibrateCamera？
        因为 stereoCalibrate 需要当前图像坐标系下的相机内参 K。
        已矫正图像的 K 不一定等于原始相机 K；
        它更接近你 undistort 时使用的 new_camera_matrix。
    """

    print(f"\n开始 {name} 单目标定 calibrateCamera...")

    if image_size is None:
        raise RuntimeError("image_size is None，无法标定")

    w, h = image_size

    # 初始内参估计：
    # fx = fy = w 是一个常见粗略初值
    # cx = w/2, cy = h/2 假设主点在图像中心附近
    init_camera_matrix = np.array(
        [
            [w, 0, w / 2.0],
            [0, w, h / 2.0],
            [0, 0, 1.0]
        ],
        dtype=np.float64
    )

    init_dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    # 对已畸变矫正图片，有两种做法：
    #
    # 做法 A：
    #   允许 OpenCV 继续估计少量残余畸变。
    #
    # 做法 B：
    #   固定畸变为 0。
    #
    # 这里采用 B，更符合“已畸变矫正图片”的假设。
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_ZERO_TANGENT_DIST |
        cv2.CALIB_FIX_K1 |
        cv2.CALIB_FIX_K2 |
        cv2.CALIB_FIX_K3 |
        cv2.CALIB_FIX_K4 |
        cv2.CALIB_FIX_K5 |
        cv2.CALIB_FIX_K6
    )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_all,
        image_points_all,
        image_size,
        init_camera_matrix,
        init_dist_coeffs,
        flags=flags
    )

    mean_error = compute_reprojection_error(
        object_points_all,
        image_points_all,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    )

    print(f"{name} RMS:", rms)
    print(f"{name} mean reprojection error:", mean_error)
    print(f"{name} camera_matrix:")
    print(camera_matrix)
    print(f"{name} dist_coeffs:")
    print(dist_coeffs)

    return rms, mean_error, camera_matrix, dist_coeffs, rvecs, tvecs


# ============================================================
# 6. 重投影误差计算
# ============================================================

def compute_reprojection_error(
    object_points_all,
    image_points_all,
    rvecs,
    tvecs,
    camera_matrix,
    dist_coeffs
):
    """
    计算平均重投影误差。

    重投影误差越小，说明标定越好。

    计算方法：
        1. 使用标定得到的 rvec/tvec/K/D 将 3D 棋盘格点投影回图像
        2. 与实际检测到的 2D 角点比较
        3. 计算平均像素误差
    """

    total_error = 0.0
    count = len(object_points_all)

    for i in range(count):
        projected_points, _ = cv2.projectPoints(
            object_points_all[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        error = cv2.norm(
            image_points_all[i],
            projected_points,
            cv2.NORM_L2
        ) / len(projected_points)

        total_error += error

    return total_error / max(1, count)


# ============================================================
# 7. stereoCalibrate
# ============================================================

def run_stereo_calibrate(
    object_points_all,
    image_points_left_all,
    image_points_right_all,
    image_size,
    K_left,
    D_left,
    K_right,
    D_right
):
    """
    使用左右角点执行 stereoCalibrate。

    输入：
        object_points_all:
            棋盘格真实 3D 点。

        image_points_left_all:
            左图 2D 角点。

        image_points_right_all:
            右图 2D 角点。

        K_left, D_left:
            左图等效内参和畸变。

        K_right, D_right:
            右图等效内参和畸变。

    输出：
        ret_stereo:
            双目标定 RMS 误差。

        R:
            右相机相对于左相机的旋转矩阵。

        T:
            右相机相对于左相机的平移向量。

        E:
            Essential Matrix。

        F:
            Fundamental Matrix。
    """

    print("\n开始 stereoCalibrate...")

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5
    )

    # 不固定内参，让 stereoCalibrate 同时优化内参和外参。
    # 因为输入是已去畸变图片，单目标定估出的 K 本身有偏差，
    # 固定它只调 R/T 会导致 RMS 偏高。
    flags = 0

    ret_stereo, K_left_out, D_left_out, K_right_out, D_right_out, R, T, E, F = cv2.stereoCalibrate(
        object_points_all,
        image_points_left_all,
        image_points_right_all,
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        criteria=criteria,
        flags=flags
    )

    print("\n================ stereoCalibrate 结果 ================")
    print("双目 RMS:")
    print(ret_stereo)

    print("\nR 右相机相对左相机旋转矩阵:")
    print(R)

    print("\nT 右相机相对左相机平移向量:")
    print(T)

    print("\nE Essential Matrix:")
    print(E)

    print("\nF Fundamental Matrix:")
    print(F)

    baseline = float(np.linalg.norm(T))
    print("\n左右相机基线长度 baseline:")
    print(f"{baseline:.6f}，单位与 square_size 一致。当前 square_size 是 mm，所以这里是 mm。")

    print("\n误差参考：")
    print("  stereo RMS < 1.0 像素：较好")
    print("  1.0 ~ 2.0 像素：一般，可以继续验证 stereoRectify")
    print("  > 2.0 像素：建议重新采集棋盘格图片或检查角点")

    return ret_stereo, K_left_out, D_left_out, K_right_out, D_right_out, R, T, E, F


# ============================================================
# 8. 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="StereoCalibrate from already-undistorted stereo chessboard images"
    )

    parser.add_argument(
        "--left-dir",
        default=DEFAULT_LEFT_DIR,
        help="左相机已畸变矫正后的棋盘格图片目录"
    )

    parser.add_argument(
        "--right-dir",
        default=DEFAULT_RIGHT_DIR,
        help="右相机已畸变矫正后的棋盘格图片目录"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="stereoCalibrate 结果保存路径 .npz"
    )

    parser.add_argument(
        "--debug-dir",
        default=DEFAULT_DEBUG_DIR,
        help="角点检测调试图保存目录"
    )

    parser.add_argument(
        "--board-cols",
        type=int,
        default=DEFAULT_BOARD_COLS,
        help="棋盘格横向内角点数量。你的 12x9 方格应填 11"
    )

    parser.add_argument(
        "--board-rows",
        type=int,
        default=DEFAULT_BOARD_ROWS,
        help="棋盘格纵向内角点数量。你的 12x9 方格应填 8"
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE,
        help="棋盘格方格边长，单位 mm。你当前是 22"
    )

    parser.add_argument(
        "--min-pairs",
        type=int,
        default=8,
        help="最少有效左右图片对数量"
    )

    parser.add_argument(
    "--check-only",
    action="store_true",
    help="只检测左右棋盘格角点并保存调试图，不执行 calibrateCamera/stereoCalibrate"
    )

    return parser.parse_args()


# ============================================================
# 9. 主函数
# ============================================================

def main():
    args = parse_args()

    print("\n================ 当前环境 ================")
    print("Python:", sys.executable)
    print("OpenCV:", cv2.__version__)
    print("cv2 path:", cv2.__file__)

    print("\n================ 当前参数 ================")
    print("left_dir    :", args.left_dir)
    print("right_dir   :", args.right_dir)
    print("output      :", args.output)
    print("debug_dir   :", args.debug_dir)
    print("board       :", args.board_cols, "x", args.board_rows, "inner corners")
    print("square_size :", args.square_size, "mm")

    # --------------------------------------------------------
    # 9.1 收集左右棋盘格角点
    # --------------------------------------------------------

    (
        object_points_all,
        image_points_left_all,
        image_points_right_all,
        image_size,
        valid_pairs
    ) = collect_stereo_chessboard_points(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        square_size=args.square_size,
        debug_dir=args.debug_dir
    )

    if args.check_only:
        print("\ncheck-only 模式：只检测棋盘格角点，不执行 stereoCalibrate。")
        print("如果 debug_corners 里的左右角点图正确，说明图片和棋盘格参数没问题。")
        return    

    if valid_pairs < args.min_pairs:
        raise RuntimeError(
            f"有效左右图片对太少: {valid_pairs}，至少需要 {args.min_pairs}。"
            f"建议采集 15~30 组。"
        )

    # --------------------------------------------------------
    # 9.2 左右分别做单目标定，得到已矫正图像坐标系下的等效 K
    # --------------------------------------------------------

    (
        rms_left,
        mean_error_left,
        K_left,
        D_left,
        rvecs_left,
        tvecs_left
    ) = calibrate_single_camera(
        object_points_all,
        image_points_left_all,
        image_size,
        name="Left"
    )

    (
        rms_right,
        mean_error_right,
        K_right,
        D_right,
        rvecs_right,
        tvecs_right
    ) = calibrate_single_camera(
        object_points_all,
        image_points_right_all,
        image_size,
        name="Right"
    )

    # --------------------------------------------------------
    # 9.3 stereoCalibrate 求双目外参 R/T
    # --------------------------------------------------------

    (
        ret_stereo,
        K_left_out,
        D_left_out,
        K_right_out,
        D_right_out,
        R,
        T,
        E,
        F
    ) = run_stereo_calibrate(
        object_points_all,
        image_points_left_all,
        image_points_right_all,
        image_size,
        K_left,
        D_left,
        K_right,
        D_right
    )

    # --------------------------------------------------------
    # 9.4 保存结果
    # --------------------------------------------------------

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.savez(
        args.output,
        image_size=np.array(image_size),
        board_cols=np.array(args.board_cols),
        board_rows=np.array(args.board_rows),
        square_size=np.array(args.square_size),
        valid_pairs=np.array(valid_pairs),

        K_left=K_left_out,
        D_left=D_left_out,
        K_right=K_right_out,
        D_right=D_right_out,

        R=R,
        T=T,
        E=E,
        F=F,

        rms_left=np.array(rms_left),
        rms_right=np.array(rms_right),
        mean_error_left=np.array(mean_error_left),
        mean_error_right=np.array(mean_error_right),
        rms_stereo=np.array(ret_stereo),

        note=np.array(
            "Computed from already-undistorted images. "
            "D_left/D_right are expected to be near zero or fixed zero."
        )
    )

    print("\n================ 保存完成 ================")
    print("stereoCalibrate 参数已保存到:")
    print(args.output)

    print("\n下一步建议：")
    print("1. 先检查 debug_corners 目录里的角点图是否全部正确。")
    print("2. 看 rms_left、rms_right、rms_stereo 是否合理。")
    print("3. 如果结果稳定，下一步再加入 stereoRectify。")


if __name__ == "__main__":
    main()