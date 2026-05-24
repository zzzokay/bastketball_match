#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dual_rknn_wide_coords_display.py

作用：
    双路摄像头实时 RKNN 人体检测 + 坐标映射 + 宽幅图画框显示。

    这个脚本不保存视频，只做实时显示和坐标输出。

整体流程：
    1. 读取左右摄像头原始 raw 图像。
    2. 左右原始视角分别送入 RKNN person 检测器。
    3. 左路检测前可使用 left_keep_mask，把左路 overlap 区域涂灰，
       这样左路只检测非重叠区域；重叠区域交给右路。
    4. 检测框坐标是 raw 原图坐标。
    5. 使用 stereo_rectify_maps_wide_good.npz 里的 raw->rectified remap 表，
       构造 raw 像素到 rectified 坐标的反查表。
    6. 根据 stitch_params_good.npz 和你的运行时微调参数：
          runtime-right-x-shift = 30
          runtime-right-y-shift = -5
       把 rectified 坐标转换到完整宽幅拼接图坐标。
    7. 构建一张完整宽幅图，并在宽幅图上画出映射后的检测框、底部中心点和坐标。

为什么不能简单缩放/平移：
    你的拼接算法做了极线矫正、旋转、变形、裁剪和右图 x/y 微调。
    所以 raw 检测框不能直接按比例映射到宽图，必须通过 map 反查 raw->rectified，
    再进入拼接坐标系。

典型运行：
    cd /home/elf/work/basketball/camera_movement

    python3 dual_rknn_wide_coords_display.py \
        --left-device /dev/video41 \
        --right-device /dev/video43 \
        --model /home/elf/work/basketball/model/basketball_player_2.1.0.rknn \
        --labels /home/elf/work/basketball/model/labels.txt \
        --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
        --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
        --width 1920 \
        --height 1080 \
        --fps 30 \
        --runtime-seam-x 150 \
        --runtime-blend-width 40 \
        --runtime-right-x-shift 30 \
        --runtime-right-y-shift -5 \
        --display-scale 0.25

按键：
    q / ESC : 退出
    s       : 保存当前宽图调试 JPG，不保存视频
"""

import argparse
import json
import os
import signal
import time
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 从第一个脚本复用 RKNN 检测器和摄像头线程。
# 两个脚本放在同一个目录 /home/elf/work/basketball/camera_movement 下即可 import。
from dual_rknn_raw_detect import (
    DEFAULT_LABELS,
    DEFAULT_LEFT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_RIGHT_DEVICE,
    FPSCounter,
    LatestFrameCamera,
    PersonDetector,
    open_camera,
)

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

STOP_REQUESTED = False
SIGINT_COUNT = 0


def handle_exit_signal(signum, frame):
    """Ctrl+C 退出处理。"""
    global STOP_REQUESTED, SIGINT_COUNT
    SIGINT_COUNT += 1
    STOP_REQUESTED = True
    if SIGINT_COUNT >= 2:
        os._exit(130)
    print("\n[信息] 收到退出信号，当前帧结束后退出")



# =============================================================================
# Async RKNN Worker：双路 RKNN 并行推理
# =============================================================================

class AsyncRKNNWorker:
    """
    单路 RKNN 异步推理线程。

    为什么要单独做这个类：
        你原来的同步写法通常是：
            left_detector.detect(left_frame)
            right_detector.detect(right_frame)

        这样左右两路是串行执行：
            左路 RKNN 约 55 ms + 右路 RKNN 约 55 ms = 总推理约 110 ms

        本类把每一路 RKNN 放到一个独立线程里：
            left_worker  在线程 A 内检测左图
            right_worker 在线程 B 内检测右图

        两个线程各自有独立的 PersonDetector / RKNNLite 实例，
        并且可以分别绑定不同 NPU core，例如 left=0，right=1。

    队列为什么 maxsize=1：
        运动场景最怕“延迟堆积”。如果 RKNN 正在处理旧帧，
        主线程又提交了新帧，本类会丢弃队列里的旧帧，只保留最新帧。
        这样检测结果虽然可能不是每帧都有更新，但不会越来越落后。
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        labels_path: str,
        obj_thresh: float,
        nms_thresh: float,
        core: str,
        use_rgb: bool = True,
    ):
        self.name = name
        self.model_path = model_path
        self.labels_path = labels_path
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.core = core
        self.use_rgb = use_rgb

        # 输入队列只存 1 帧：永远处理最新帧，避免延迟堆积。
        self.input_queue = queue.Queue(maxsize=1)

        # latest_* 是 worker 最近一次完成的检测结果。
        self.lock = threading.Lock()
        self.latest_frame_id = -1
        self.latest_results = []
        self.latest_infer_ms = 0.0
        self.total_infer_count = 0

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.detector = None

    def start(self):
        """启动后台 RKNN 推理线程。"""
        self.thread.start()
        return self

    def submit(self, frame_id: int, frame_bgr: np.ndarray) -> None:
        """
        向 worker 提交一帧图像。

        注意：
            这里不主动 copy 图像，减少 1920x1080 大图复制开销。
            OpenCV 后台采集线程每次 read 通常会返回新的 ndarray，正常可用。
        """
        if frame_bgr is None:
            return

        item = (int(frame_id), frame_bgr)

        try:
            self.input_queue.put_nowait(item)
        except queue.Full:
            # 如果队列满了，说明上一帧还没开始处理。
            # 这里丢掉旧帧，放入最新帧。
            try:
                _ = self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(item)
            except queue.Full:
                pass

    def get_latest(self):
        """
        获取最近一次完成的检测结果。

        返回：
            det_frame_id:
                该检测结果对应的输入帧编号。

            results:
                PersonDetector.detect() 输出：
                [(class_id, score, (x1, y1, x2, y2)), ...]

            infer_ms:
                最近一次推理耗时。
        """
        with self.lock:
            return (
                int(self.latest_frame_id),
                list(self.latest_results),
                float(self.latest_infer_ms),
            )

    def _run(self) -> None:
        """
        worker 线程主函数。

        关键点：PersonDetector 必须在 worker 内部创建。
        这样左右两个 worker 才是两个独立 RKNNLite 实例，避免共享 runtime。
        """
        try:
            self.detector = PersonDetector(
                model_path=self.model_path,
                labels_path=self.labels_path,
                obj_thresh=self.obj_thresh,
                nms_thresh=self.nms_thresh,
                use_rgb=self.use_rgb,
                core=self.core,
                name=self.name,
            )

            while not self.stop_event.is_set():
                try:
                    frame_id, frame = self.input_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                t0 = time.perf_counter()
                results = self.detector.detect(frame)
                t1 = time.perf_counter()

                with self.lock:
                    self.latest_frame_id = frame_id
                    self.latest_results = results
                    self.latest_infer_ms = (t1 - t0) * 1000.0
                    self.total_infer_count += 1

        except Exception as e:
            print(f"[{self.name}] RKNN worker 异常: {e}")

        finally:
            if self.detector is not None:
                self.detector.close()

    def stop(self) -> None:
        """停止 worker，并释放 RKNN 资源。"""
        self.stop_event.set()
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass

# =============================================================================
# 1. 数据结构
# =============================================================================

@dataclass
class RectifyMaps:
    """
    raw -> rectified 的 remap 查找表。

    注意：
        OpenCV remap 的 map 是“输出坐标到输入坐标”的映射：
            rectified(x, y) = raw(map_x[y, x], map_y[y, x])

        你的需求是：
            raw 检测点 -> rectified 点 -> wide 点

        所以本脚本会根据 map_x/map_y 额外构造一个 raw -> rectified 反查表。
    """
    raw_image_size: Tuple[int, int]
    rectified_size: Tuple[int, int]
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


@dataclass
class StitchParams:
    """拼接参数，来自 stitch_params_good.npz。"""
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


@dataclass
class WideDetection:
    """一个映射到宽幅图上的检测结果。"""
    source: str
    score: float
    raw_bbox: Tuple[int, int, int, int]
    rect_bottom: Tuple[float, float]
    wide_bottom: Tuple[float, float]
    wide_bbox: Tuple[float, float, float, float]


# =============================================================================
# 2. npz 加载与基础工具
# =============================================================================


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    """从 npz 字段中读取 [width, height]。"""
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 格式错误: {value}")
    return int(flat[0]), int(flat[1])


def get_npz_int(data: np.lib.npyio.NpzFile, key: str, default: Optional[int] = None) -> int:
    """从 npz 中读取 int 字段。"""
    if key not in data.files:
        if default is None:
            raise RuntimeError(f"npz 缺少字段: {key}")
        return int(default)
    return int(np.array(data[key]).reshape(-1)[0])


def to_float_xy_maps(map1: np.ndarray, map2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 OpenCV remap map 转成两个 float32 数组：map_x, map_y。

    常见情况：
        map1: HxW float32，表示 raw_x
        map2: HxW float32，表示 raw_y

    如果 map 已经是 CV_16SC2 fixed-point 格式，本函数会退化使用整数部分，
    对坐标反查精度略低，但通常你的 npz 文件保存的是 float map。
    """
    if map1.ndim == 3 and map1.shape[2] >= 2:
        map_x = map1[:, :, 0].astype(np.float32)
        map_y = map1[:, :, 1].astype(np.float32)
    else:
        map_x = map1.astype(np.float32)
        map_y = map2.astype(np.float32)
    return map_x, map_y


def load_rectify_maps(map_file: str) -> RectifyMaps:
    """加载 stereo_rectify_maps_wide_good.npz。"""
    if not os.path.exists(map_file):
        raise RuntimeError(f"找不到 map 文件: {map_file}")

    data = np.load(map_file)
    required = ["left_rect_map1", "left_rect_map2", "right_rect_map1", "right_rect_map2", "raw_image_size"]
    for key in required:
        if key not in data.files:
            raise RuntimeError(f"map 文件缺少字段: {key}")

    raw_image_size = parse_image_size(data["raw_image_size"])
    if "rectified_size" in data.files:
        rectified_size = parse_image_size(data["rectified_size"])
    else:
        rectified_size = (int(data["left_rect_map1"].shape[1]), int(data["left_rect_map1"].shape[0]))

    maps = RectifyMaps(
        raw_image_size=raw_image_size,
        rectified_size=rectified_size,
        left_map1=data["left_rect_map1"],
        left_map2=data["left_rect_map2"],
        right_map1=data["right_rect_map1"],
        right_map2=data["right_rect_map2"],
    )

    print("[信息] 已加载 remap 文件")
    print(f"  map_file       : {map_file}")
    print(f"  raw_image_size : {maps.raw_image_size}")
    print(f"  rectified_size : {maps.rectified_size}")
    return maps


def load_stitch_params(stitch_param_file: str) -> StitchParams:
    """加载 stitch_params_good.npz。"""
    if not os.path.exists(stitch_param_file):
        raise RuntimeError(f"找不到 stitch 参数文件: {stitch_param_file}")

    data = np.load(stitch_param_file)
    overlap_px = get_npz_int(data, "overlap_px")

    params = StitchParams(
        overlap_px=overlap_px,
        vertical_offset=get_npz_int(data, "vertical_offset"),
        blend_width=get_npz_int(data, "blend_width", default=overlap_px),
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

    print("[信息] 已加载拼接参数")
    print(f"  overlap_px      : {params.overlap_px}")
    print(f"  vertical_offset : {params.vertical_offset}")
    print(f"  output_size     : {params.output_width} x {params.output_height}")
    print(f"  left_y          : {params.left_y1} -> {params.left_y2}")
    print(f"  right_y         : {params.right_y1} -> {params.right_y2}")
    return params


# =============================================================================
# 3. 构造 raw -> rectified 反查表
# =============================================================================


def build_raw_to_rect_inverse_map(
    map1: np.ndarray,
    map2: np.ndarray,
    raw_size: Tuple[int, int],
    cache_file: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 rectified->raw remap 表，构造 raw->rectified 反查表。

    OpenCV remap 表含义：
        对每个 rectified 输出像素 (rx, ry)，map 中保存它来自 raw 的哪个位置：
            raw_x = map_x[ry, rx]
            raw_y = map_y[ry, rx]

    我们需要反过来查：
        某个 raw 检测点 (raw_x, raw_y) 对应到 rectified 哪里？

    做法：
        遍历所有 rectified 像素，把它们对应的 raw 坐标四舍五入到整数，
        然后在 inv_x[raw_y, raw_x] / inv_y[raw_y, raw_x] 里保存 rectified 坐标。

    说明：
        反查表不是严格一一对应，会有少量空洞，所以 lookup 时还会做邻域搜索。
    """
    if cache_file and os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            print(f"[信息] 读取 raw->rectified 反查缓存: {cache_file}")
            return data["inv_x"].astype(np.float32), data["inv_y"].astype(np.float32)
        except Exception as e:
            print(f"[警告] 读取缓存失败，将重新生成: {e}")

    raw_w, raw_h = raw_size
    map_x, map_y = to_float_xy_maps(map1, map2)
    rect_h, rect_w = map_x.shape[:2]

    inv_x = np.full((raw_h, raw_w), np.nan, dtype=np.float32)
    inv_y = np.full((raw_h, raw_w), np.nan, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(rect_w, dtype=np.float32), np.arange(rect_h, dtype=np.float32))

    raw_x = np.rint(map_x).astype(np.int32)
    raw_y = np.rint(map_y).astype(np.int32)

    valid = (raw_x >= 0) & (raw_x < raw_w) & (raw_y >= 0) & (raw_y < raw_h)
    inv_x[raw_y[valid], raw_x[valid]] = grid_x[valid]
    inv_y[raw_y[valid], raw_x[valid]] = grid_y[valid]

    valid_ratio = float(np.isfinite(inv_x).mean())
    print(f"[信息] raw->rectified 反查表生成完成，有效覆盖率: {valid_ratio * 100:.1f}%")

    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez_compressed(cache_file, inv_x=inv_x, inv_y=inv_y)
        print(f"[信息] 已保存反查缓存: {cache_file}")

    return inv_x, inv_y


def lookup_rect_point(
    inv_x: np.ndarray,
    inv_y: np.ndarray,
    raw_x: float,
    raw_y: float,
    search_radius: int = 5,
) -> Optional[Tuple[float, float]]:
    """
    根据 raw 坐标查 rectified 坐标。

    如果正好的 raw 像素没有反查值，就在周围 search_radius 范围内找最近的有效点。
    """
    h, w = inv_x.shape[:2]
    ix = int(round(raw_x))
    iy = int(round(raw_y))

    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None

    if np.isfinite(inv_x[iy, ix]) and np.isfinite(inv_y[iy, ix]):
        return float(inv_x[iy, ix]), float(inv_y[iy, ix])

    x1 = max(0, ix - search_radius)
    x2 = min(w, ix + search_radius + 1)
    y1 = max(0, iy - search_radius)
    y2 = min(h, iy + search_radius + 1)

    patch_valid = np.isfinite(inv_x[y1:y2, x1:x2]) & np.isfinite(inv_y[y1:y2, x1:x2])
    if not np.any(patch_valid):
        return None

    yy, xx = np.where(patch_valid)
    xx_abs = xx + x1
    yy_abs = yy + y1
    dist2 = (xx_abs - ix) ** 2 + (yy_abs - iy) ** 2
    best = int(np.argmin(dist2))
    bx = int(xx_abs[best])
    by = int(yy_abs[best])
    return float(inv_x[by, bx]), float(inv_y[by, bx])


# =============================================================================
# 4. 拼接坐标系映射
# =============================================================================


def rect_point_to_wide(
    source: str,
    rect_x: float,
    rect_y: float,
    params: StitchParams,
    right_x_shift: int,
    right_y_shift: int,
) -> Optional[Tuple[float, float]]:
    """
    把 rectified 坐标转换成完整宽幅图坐标。

    宽幅图结构：
        [left_keep][overlap][right_keep]

    左图：
        只接收 left_keep 区域。
        因为你的策略是：左图 overlap 直接去掉，不在左 overlap 检测。

    右图：
        负责 overlap + right_keep。
        注意你的实时拼接有 right_x_shift / right_y_shift 微调，
        所以从右 rectified 坐标转宽图时要反向扣掉这两个偏移。
    """
    left_keep_w = params.left_keep_x2 - params.left_keep_x1
    overlap_w = params.overlap_px

    overlap_start = left_keep_w
    right_keep_start = left_keep_w + overlap_w

    if source == "left":
        # 左图只保留 left_keep，不保留 left_overlap。
        if not (params.left_keep_x1 <= rect_x < params.left_keep_x2):
            return None
        if not (params.left_y1 <= rect_y < params.left_y2):
            return None
        wide_x = rect_x - params.left_keep_x1
        wide_y = rect_y - params.left_y1
        return float(wide_x), float(wide_y)

    if source == "right":
        # 右图拼接时使用了 right_x_shift/right_y_shift。
        # 输出 wide 的某个右图像素，对应的采样 rect_x 是：
        #   params.right_overlap_x1 + local_x + right_x_shift
        # 所以反向映射时 local_x = rect_x - right_x_shift - base_x。
        corrected_x = rect_x - right_x_shift
        corrected_y = rect_y - right_y_shift

        if not (params.right_y1 <= corrected_y < params.right_y2):
            return None

        if params.right_overlap_x1 <= corrected_x < params.right_overlap_x2:
            local_x = corrected_x - params.right_overlap_x1
            wide_x = overlap_start + local_x
            wide_y = corrected_y - params.right_y1
            return float(wide_x), float(wide_y)

        if params.right_keep_x1 <= corrected_x < params.right_keep_x2:
            local_x = corrected_x - params.right_keep_x1
            wide_x = right_keep_start + local_x
            wide_y = corrected_y - params.right_y1
            return float(wide_x), float(wide_y)

        return None

    raise ValueError(f"未知 source: {source}")


def bbox_sample_points(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[float, float]]:
    """
    从 raw 检测框上取多个采样点。

    由于矫正/旋转/变形不是简单线性关系，只映射左上和右下可能不准。
    这里取四角、边中点、中心点、底部中心点，然后映射后求外接框。
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return [
        (x1, y1), (cx, y1), (x2, y1),
        (x1, cy), (cx, cy), (x2, cy),
        (x1, y2), (cx, y2), (x2, y2),
    ]


def map_raw_detection_to_wide(
    source: str,
    score: float,
    bbox: Tuple[int, int, int, int],
    inv_x: np.ndarray,
    inv_y: np.ndarray,
    params: StitchParams,
    right_x_shift: int,
    right_y_shift: int,
) -> Optional[WideDetection]:
    """
    把一个 raw 检测框映射到完整宽幅图。

    返回 None 表示该检测框不属于当前 source 应负责的拼接区域，或者映射失败。
    """
    x1, y1, x2, y2 = bbox

    # 1. 人物底部中心点：后续运镜最应该使用这个点。
    raw_bottom_x = (x1 + x2) / 2.0
    raw_bottom_y = float(y2)
    rect_bottom = lookup_rect_point(inv_x, inv_y, raw_bottom_x, raw_bottom_y)
    if rect_bottom is None:
        return None

    wide_bottom = rect_point_to_wide(source, rect_bottom[0], rect_bottom[1], params, right_x_shift, right_y_shift)
    if wide_bottom is None:
        return None

    # 2. 检测框边界点映射，用于在宽图上画外接框。
    wide_points = []
    for px, py in bbox_sample_points(x1, y1, x2, y2):
        rect_p = lookup_rect_point(inv_x, inv_y, px, py)
        if rect_p is None:
            continue
        wide_p = rect_point_to_wide(source, rect_p[0], rect_p[1], params, right_x_shift, right_y_shift)
        if wide_p is not None:
            wide_points.append(wide_p)

    # 如果框采样点太少，至少用底部中心构造一个很小的框，避免整个检测丢掉。
    if len(wide_points) < 2:
        bx, by = wide_bottom
        wide_bbox = (bx - 5.0, by - 5.0, bx + 5.0, by + 5.0)
    else:
        xs = [p[0] for p in wide_points]
        ys = [p[1] for p in wide_points]
        wide_bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

    return WideDetection(
        source=source,
        score=float(score),
        raw_bbox=(int(x1), int(y1), int(x2), int(y2)),
        rect_bottom=(float(rect_bottom[0]), float(rect_bottom[1])),
        wide_bottom=(float(wide_bottom[0]), float(wide_bottom[1])),
        wide_bbox=wide_bbox,
    )


# =============================================================================
# 5. 左路 keep mask：左 overlap 不检测
# =============================================================================


def build_left_keep_raw_mask(maps: RectifyMaps, params: StitchParams, dilate_iter: int = 2) -> np.ndarray:
    """
    根据左图 rectified 的 left_keep 区域，生成 raw 坐标系 mask。

    目的：
        你的策略是：重叠区域直接去掉左边，只让右图负责 overlap。
        所以左路送入 RKNN 前，把不属于 left_keep 的 raw 像素涂成灰色，
        让左路不会检测 overlap 里的人员，避免左右重复输出。
    """
    raw_w, raw_h = maps.raw_image_size
    mask = np.zeros((raw_h, raw_w), dtype=np.uint8)

    map_x, map_y = to_float_xy_maps(maps.left_map1, maps.left_map2)

    # 在 rectified 坐标系里取 left_keep 的有效区域。
    roi_x1 = params.left_keep_x1
    roi_x2 = params.left_keep_x2
    roi_y1 = params.left_y1
    roi_y2 = params.left_y2

    raw_x = np.rint(map_x[roi_y1:roi_y2, roi_x1:roi_x2]).astype(np.int32)
    raw_y = np.rint(map_y[roi_y1:roi_y2, roi_x1:roi_x2]).astype(np.int32)

    valid = (raw_x >= 0) & (raw_x < raw_w) & (raw_y >= 0) & (raw_y < raw_h)
    mask[raw_y[valid], raw_x[valid]] = 255

    # remap 反投影后会有稀疏孔洞，膨胀几次，让 mask 更连续。
    if dilate_iter > 0:
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask


def apply_mask_for_detection(frame: np.ndarray, mask: np.ndarray, fill_value: int = 114) -> np.ndarray:
    """把 mask 外区域涂灰，用于 RKNN 检测前屏蔽不希望检测的区域。"""
    out = frame.copy()
    out[mask == 0] = (fill_value, fill_value, fill_value)
    return out


# =============================================================================
# 6. 构建完整宽幅图，用于显示画框
# =============================================================================


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
    从 raw 图 remap 出 rectified 坐标系下的 ROI。

    返回固定大小。如果 ROI 越界，越界部分用黑色补齐。
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
    生成完整宽幅图，只用于显示检测结果。

    这里和你前面调通的拼接逻辑一致：
        left_keep + seam 附近小范围融合 + right overlap/right_keep
    并且加入了 runtime-right-x-shift / runtime-right-y-shift 微调。
    """
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
    overlap_start = left_keep_w
    right_keep_start = left_keep_w + overlap_w

    left_y1 = params.left_y1
    left_y2 = left_y1 + output_h
    right_y1 = params.right_y1 + right_y_shift
    right_y2 = right_y1 + output_h

    # 1. 左侧非重叠区域。
    left_keep = remap_rectified_roi_fixed_size(
        left_raw,
        maps.left_map1,
        maps.left_map2,
        params.left_keep_x1,
        left_y1,
        params.left_keep_x2,
        left_y2,
    )
    wide[:, left_keep_start:left_keep_start + left_keep_w] = left_keep

    # 2. overlap：接缝左边取左图，接缝右边取右图，中间小范围融合。
    blend_width = max(1, min(int(blend_width), overlap_w))
    if seam_x < 0:
        seam_x_used = overlap_w // 2
    else:
        seam_x_used = int(seam_x)

    half = blend_width // 2
    seam_x_used = int(np.clip(seam_x_used, half, overlap_w - (blend_width - half)))
    blend_x1 = seam_x_used - half
    blend_x2 = blend_x1 + blend_width

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
        wide[:, overlap_start + blend_x2:overlap_start + overlap_w] = right_overlap_right

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
    blend = left_blend.astype(np.float32) * (1.0 - alpha) + right_blend.astype(np.float32) * alpha
    wide[:, overlap_start + blend_x1:overlap_start + blend_x2] = np.clip(blend, 0, 255).astype(np.uint8)

    # 3. 右侧非重叠区域。
    right_keep = remap_rectified_roi_fixed_size(
        right_raw,
        maps.right_map1,
        maps.right_map2,
        params.right_keep_x1 + right_x_shift,
        right_y1,
        params.right_keep_x2 + right_x_shift,
        right_y2,
    )
    wide[:, right_keep_start:right_keep_start + right_keep_w] = right_keep

    return wide


# =============================================================================
# 7. 画框、坐标输出
# =============================================================================


def draw_wide_detections(wide: np.ndarray, detections: List[WideDetection]) -> np.ndarray:
    """在完整宽幅图上画检测框、底部中心点和坐标。"""
    vis = wide.copy()

    for det in detections:
        x1, y1, x2, y2 = det.wide_bbox
        bx, by = det.wide_bottom

        color = (0, 255, 0) if det.source == "right" else (255, 160, 0)

        x1i = int(round(x1)); y1i = int(round(y1))
        x2i = int(round(x2)); y2i = int(round(y2))
        bxi = int(round(bx)); byi = int(round(by))

        cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2)
        cv2.circle(vis, (bxi, byi), 6, (0, 0, 255), -1)

        label = f"{det.source} person {det.score:.2f} bottom=({bxi},{byi})"
        cv2.putText(vis, label, (x1i, max(30, y1i - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return vis


def wide_detections_to_jsonable(frame_id: int, detections: List[WideDetection]) -> Dict:
    """把宽幅检测结果转换成 JSON 可写格式。"""
    return {
        "frame": int(frame_id),
        "time": time.time(),
        "count": len(detections),
        "detections": [
            {
                "source": det.source,
                "score": det.score,
                "raw_bbox": list(det.raw_bbox),
                "rect_bottom": [det.rect_bottom[0], det.rect_bottom[1]],
                "wide_bottom": [det.wide_bottom[0], det.wide_bottom[1]],
                "wide_bbox": [det.wide_bbox[0], det.wide_bbox[1], det.wide_bbox[2], det.wide_bbox[3]],
            }
            for det in detections
        ],
    }


def print_wide_detections(frame_id: int, detections: List[WideDetection]) -> None:
    """终端打印当前帧每个人的宽幅坐标。"""
    if not detections:
        print(f"[FRAME {frame_id}] no person")
        return

    print(f"[FRAME {frame_id}] persons={len(detections)}")
    for i, det in enumerate(detections):
        bx, by = det.wide_bottom
        x1, y1, x2, y2 = det.wide_bbox
        print(
            f"  #{i} {det.source:5s} conf={det.score:.2f} "
            f"wide_bottom=({bx:.1f}, {by:.1f}) "
            f"wide_bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )


# =============================================================================
# 8. 主流程
# =============================================================================


def main():
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    parser = argparse.ArgumentParser(description="双路 RKNN 检测 -> 宽幅坐标映射 -> 宽图画框显示")

    # 摄像头与 RKNN 参数。
    parser.add_argument("--left-device", default=DEFAULT_LEFT_DEVICE, help="左摄像头设备，例如 /dev/video41")
    parser.add_argument("--right-device", default=DEFAULT_RIGHT_DEVICE, help="右摄像头设备，例如 /dev/video43")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="RKNN 模型路径")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="labels.txt 路径")
    parser.add_argument("--width", type=int, default=1920, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=30, help="摄像头采集 FPS")
    parser.add_argument("--conf", type=float, default=0.25, help="person 置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS 阈值")
    parser.add_argument("--left-core", default="0", help="左路 RKNN NPU core：0/1/2/all/auto")
    parser.add_argument("--right-core", default="1", help="右路 RKNN NPU core：0/1/2/all/auto")
    parser.add_argument("--bgr-input", action="store_true", help="模型输入为 BGR；默认 BGR->RGB")

    # map/stitch 参数。
    parser.add_argument("--map-file", default="/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz")
    parser.add_argument("--stitch-param", default="/home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz")
    parser.add_argument("--runtime-seam-x", type=int, default=150, help="运行时接缝位置")
    parser.add_argument("--runtime-blend-width", type=int, default=40, help="运行时融合宽度")
    parser.add_argument("--runtime-right-x-shift", type=int, default=30, help="右图 x 微调")
    parser.add_argument("--runtime-right-y-shift", type=int, default=-5, help="右图 y 微调")
    parser.add_argument("--inverse-cache-dir", default="/home/elf/work/basketball/camera_movement/cache", help="raw->rectified 反查缓存目录")

    # 检测区域与输出。
    parser.add_argument("--disable-left-keep-mask", action="store_true", help="禁用左路 keep mask；默认左路不检测 overlap")
    parser.add_argument("--output-jsonl", default="", help="可选：保存每帧宽幅坐标 JSONL，不保存视频")
    parser.add_argument("--print-every", type=int, default=10, help="每 N 帧打印一次坐标，0 表示不打印")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0 表示无限")

    # 显示参数。
    parser.add_argument("--display-scale", type=float, default=0.25, help="宽幅图显示缩放比例")
    parser.add_argument("--headless", action="store_true", help="不显示窗口，只打印/输出坐标")
    parser.add_argument("--save-debug-dir", default="/home/elf/work/basketball/camera_movement/debug_wide", help="按 s 保存调试图目录")

    args = parser.parse_args()

    maps = load_rectify_maps(args.map_file)
    params = load_stitch_params(args.stitch_param)

    print("[信息] 运行时微调参数")
    print(f"  seam_x       : {args.runtime_seam_x}")
    print(f"  blend_width  : {args.runtime_blend_width}")
    print(f"  right_x_shift: {args.runtime_right_x_shift}")
    print(f"  right_y_shift: {args.runtime_right_y_shift}")

    os.makedirs(args.inverse_cache_dir, exist_ok=True)
    left_cache = os.path.join(args.inverse_cache_dir, "left_raw_to_rect_inverse.npz")
    right_cache = os.path.join(args.inverse_cache_dir, "right_raw_to_rect_inverse.npz")

    print("[信息] 准备 left raw->rectified 反查表")
    left_inv_x, left_inv_y = build_raw_to_rect_inverse_map(maps.left_map1, maps.left_map2, maps.raw_image_size, left_cache)
    print("[信息] 准备 right raw->rectified 反查表")
    right_inv_x, right_inv_y = build_raw_to_rect_inverse_map(maps.right_map1, maps.right_map2, maps.raw_image_size, right_cache)

    left_keep_mask = None
    if not args.disable_left_keep_mask:
        print("[信息] 生成左路 keep mask：左路只检测非重叠区域")
        left_keep_mask = build_left_keep_raw_mask(maps, params)
        mask_path = os.path.join(args.inverse_cache_dir, "left_keep_mask.png")
        cv2.imwrite(mask_path, left_keep_mask)
        print(f"[信息] left_keep_mask 已保存: {mask_path}")

    left_worker = None
    right_worker = None
    left_cam = None
    right_cam = None
    jsonl_fp = None

    try:
        # 左右两个 RKNN worker 在后台线程中各自创建独立 RKNNLite runtime。
        # 这里不要再在主线程里创建 left_detector/right_detector。
        left_worker = AsyncRKNNWorker(
            name="left-rknn",
            model_path=args.model,
            labels_path=args.labels,
            obj_thresh=args.conf,
            nms_thresh=args.nms,
            core=args.left_core,
            use_rgb=not args.bgr_input,
        ).start()
        right_worker = AsyncRKNNWorker(
            name="right-rknn",
            model_path=args.model,
            labels_path=args.labels,
            obj_thresh=args.conf,
            nms_thresh=args.nms,
            core=args.right_core,
            use_rgb=not args.bgr_input,
        ).start()

        cap_left = open_camera(args.left_device, args.width, args.height, args.fps)
        cap_right = open_camera(args.right_device, args.width, args.height, args.fps)
        left_cam = LatestFrameCamera(cap_left, "left").start()
        right_cam = LatestFrameCamera(cap_right, "right").start()

        if args.output_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
            jsonl_fp = open(args.output_jsonl, "w", encoding="utf-8")
            print(f"[输出] 宽幅坐标 JSONL: {args.output_jsonl}")

        fps_counter = FPSCounter()
        frame_id = 0
        debug_id = 0

        print("[信息] 开始：双路异步 RKNN -> 宽幅坐标 -> 宽图画框。按 q/ESC 退出，s 保存调试图。")

        while not STOP_REQUESTED:
            ok_l, left_raw = left_cam.read_latest()
            ok_r, right_raw = right_cam.read_latest()
            if not ok_l or not ok_r:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()

            # 左路默认屏蔽 overlap，只保留 left_keep 区域检测。
            if left_keep_mask is not None:
                left_for_detect = apply_mask_for_detection(left_raw, left_keep_mask)
            else:
                left_for_detect = left_raw

            # ------------------------------------------------------------
            # 左右 RKNN 并行推理
            # ------------------------------------------------------------
            # submit() 只把当前帧交给后台线程，不等待推理完成。
            # get_latest() 读取最近一次已经完成的结果。
            # 因此 wide 框可能会比当前画面慢 1~2 帧，这是异步推理的正常现象。
            left_worker.submit(frame_id, left_for_detect)
            right_worker.submit(frame_id, right_raw)

            left_det_frame, left_results, left_infer_ms = left_worker.get_latest()
            right_det_frame, right_results, right_infer_ms = right_worker.get_latest()

            # raw 检测结果 -> 宽幅坐标。
            wide_dets: List[WideDetection] = []

            for _, score, bbox in left_results:
                mapped = map_raw_detection_to_wide(
                    source="left",
                    score=score,
                    bbox=bbox,
                    inv_x=left_inv_x,
                    inv_y=left_inv_y,
                    params=params,
                    right_x_shift=args.runtime_right_x_shift,
                    right_y_shift=args.runtime_right_y_shift,
                )
                if mapped is not None:
                    wide_dets.append(mapped)

            for _, score, bbox in right_results:
                mapped = map_raw_detection_to_wide(
                    source="right",
                    score=score,
                    bbox=bbox,
                    inv_x=right_inv_x,
                    inv_y=right_inv_y,
                    params=params,
                    right_x_shift=args.runtime_right_x_shift,
                    right_y_shift=args.runtime_right_y_shift,
                )
                if mapped is not None:
                    wide_dets.append(mapped)

            # 用当前 raw 图构建宽幅图，并把检测框画上去。
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
            vis = draw_wide_detections(wide, wide_dets)

            t1 = time.perf_counter()
            fps = fps_counter.update()

            cv2.putText(
                vis,
                (
                    f"FPS:{fps:.1f} persons:{len(wide_dets)} "
                    f"total:{(t1-t0)*1000:.1f}ms "
                    f"L:{left_infer_ms:.1f}ms@{left_det_frame} "
                    f"R:{right_infer_ms:.1f}ms@{right_det_frame}"
                ),
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            if args.output_jsonl:
                jsonl_fp.write(json.dumps(wide_detections_to_jsonable(frame_id, wide_dets), ensure_ascii=False) + "\n")
                jsonl_fp.flush()

            if args.print_every > 0 and frame_id % args.print_every == 0:
                print_wide_detections(frame_id, wide_dets)

            if not args.headless:
                show = vis
                if args.display_scale != 1.0:
                    show = cv2.resize(show, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
                cv2.imshow("RKNN wide coordinates", show)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    os.makedirs(args.save_debug_dir, exist_ok=True)
                    path = os.path.join(args.save_debug_dir, f"wide_detect_{debug_id:06d}.jpg")
                    cv2.imwrite(path, vis)
                    print(f"[保存] {path}")
                    debug_id += 1

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()
        if left_cam is not None:
            left_cam.stop()
        if right_cam is not None:
            right_cam.stop()
        try:
            cap_left.release()
            cap_right.release()
        except Exception:
            pass
        if left_worker is not None:
            left_worker.stop()
        if right_worker is not None:
            right_worker.stop()
        cv2.destroyAllWindows()
        print("[信息] 程序退出")


if __name__ == "__main__":
    main()
