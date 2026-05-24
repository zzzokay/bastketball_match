#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dual_rknn_director_view1920_fast.py

用途：
    RK3588 / ELF2 上的“检测低频 + 运镜高频 + 直接输出 1920x1080 视口”脚本。

核心目标：
    1. 双路摄像头后台采集，主循环不等待 cap.read()。
    2. 单 RKNN 交替检测左右路，避免双路 RKNN 抢 NPU。
    3. 检测框从 raw 原图坐标映射到完整宽幅坐标。
    4. 把宽幅坐标送入你的 predict1_weighted / predict1_director 运镜逻辑。
    5. 不生成完整宽幅图；只根据运镜得到的 crop_x/crop_y，直接 remap 出 1920x1080 view。
    6. 目标是“最终运镜画面 20 FPS 以上”，检测帧率可以低于输出帧率。

为什么这样才能快：
    你之前的日志里，RKNN 单次推理约 55ms；如果每帧都检测，理论上不可能超过 18 FPS。
    但是运镜输出不需要每一帧都重新检测：
        - RKNN 每隔 N 帧更新一次目标坐标；
        - 中间帧继续使用最近一次检测结果；
        - 导播状态每帧更新；
        - 1920x1080 view 每帧直接从 raw -> rectified ROI remap 生成。

依赖文件：
    请把本脚本放在：
        /home/elf/work/basketball/camera_movement

    同目录下需要有：
        dual_rknn_wide_coords_display_alternating.py
        predict1_weighted.py
        predict1_director.py

    其中 dual_rknn_wide_coords_display_alternating.py 复用你已经跑通的：
        PersonDetector
        map / stitch 参数加载
        raw 检测框 -> wide 坐标映射
        left_keep_mask
        SmoothTracks

典型运行：
    cd /home/elf/work/basketball/camera_movement

python3 dual_rknn_director_view1920_fast.py \
    --left-device /dev/video41 \
    --right-device /dev/video43 \
    --model /home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn \
    --labels /home/elf/work/basketball/model/labels.txt \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
    --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --conf 0.25 \
    --nms 0.45 \
    --rknn-core -1 \
    --runtime-seam-x 150 \
    --runtime-blend-width 40 \
    --runtime-right-x-shift 30 \
    --runtime-right-y-shift -5 \
    --detect-interval 3 \
    --smooth 0.70 \
    --view-width 1920 \
    --view-height 1080 \
    --crop-y-mode center \
    --display-scale 0.5 \
    --print-every 30

    
按键：
    q / ESC : 退出
    s       : 保存当前 1920x1080 view 调试 JPG
    l       : 开启/关闭左路 overlap 屏蔽

重要建议：
    如果你只要最终输出 20 FPS 以上：
        --detect-interval 3 或 4 更现实。
    如果你强行 --detect-interval 1，单次 RKNN 55ms 会限制整体帧率。
"""

import argparse
import os
import sys
import time
import signal
import threading
import select
import types
from pathlib import Path
from typing import List, Tuple, Optional

# RK3588 / Mali 上禁用 OpenCL，避免 OpenCV 额外开销或 CL_INVALID_BINARY。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# -----------------------------------------------------------------------------
# 让 predict1_director.py 即使找不到 predict1_yolo.py 也能导入。
# 你的 predict1_director.py 顶部有：
#     from predict1_yolo import draw_boxes_on_frame
# 本脚本不需要那个函数，所以提供一个空实现作为兜底。
# -----------------------------------------------------------------------------
if "predict1_yolo" not in sys.modules:
    fake_yolo = types.ModuleType("predict1_yolo")

    def _draw_boxes_on_frame_stub(frame, boxes_info):
        return frame

    fake_yolo.draw_boxes_on_frame = _draw_boxes_on_frame_stub
    sys.modules["predict1_yolo"] = fake_yolo

# -----------------------------------------------------------------------------
# 复用你已经跑通的基础模块。
# 这个模块来自之前的 dual_rknn_wide_coords_display_alternating.py。
# -----------------------------------------------------------------------------
try:
    import dual_rknn_wide_coords_display_alternating as base
except Exception as e:
    print("[错误] 无法导入 dual_rknn_wide_coords_display_alternating.py")
    print("       请确认本脚本和它在同一个目录。")
    raise

# -----------------------------------------------------------------------------
# 导入你的运镜算法。
# predict1_weighted.py 负责人物框分析、主战区、热点等。
# predict1_director.py 负责导播状态、窗口切换、锚点惯性等。
# -----------------------------------------------------------------------------
try:
    from predict1_weighted import (
        init_single_view_state,
        analyze_single_view_frame,
        VERTICAL_HOME_Y_RATIO,
        BASE_CAMERA_SCALE,
        clamp,
    )
    from predict1_director import (
        init_overlay_director_state,
        update_overlay_director_state,
    )
except Exception as e:
    print("[错误] 无法导入 predict1_weighted.py / predict1_director.py")
    print("       请确认这两个文件在 /home/elf/work/basketball/camera_movement 目录下。")
    raise


STOP_REQUESTED = False


def handle_exit_signal(signum, frame):
    """Ctrl+C / kill 时设置退出标志，尽量释放摄像头和 RKNN。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[信息] 收到退出信号，当前帧结束后退出。")


# =============================================================================
# 1. 后台摄像头采集
# =============================================================================

class LatestFrameCamera:
    """
    后台采集摄像头。

    为什么需要：
        同步 cap.read() 可能阻塞主循环，你前面测试 total 从 145ms 降到 120ms，
        说明摄像头读取确实占了一部分时间。

    设计：
        - 每个摄像头一个线程；
        - 线程持续 read 最新帧；
        - 主循环只取 latest_frame，不等待硬件采集；
        - 如果主循环处理慢，旧帧自动被新帧覆盖，避免延迟堆积。
    """

    def __init__(self, device: str, width: int, height: int, fps: int, name: str):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name

        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_index = -1
        self.latest_time = 0.0
        self.read_fail_count = 0

    def start(self):
        self.cap = base.open_camera(self.device, self.width, self.height, self.fps)
        self.thread = threading.Thread(target=self._loop, name=f"{self.name}-capture", daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        idx = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.read_fail_count += 1
                time.sleep(0.005)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_index = idx
                self.latest_time = time.time()
            idx += 1

    def wait_first_frame(self, timeout: float = 3.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.latest_frame is not None:
                    return True
            time.sleep(0.01)
        return False

    def get_latest(self):
        """
        返回：index, frame, timestamp。
        不 copy 大图，减少 1920x1080 内存复制开销。
        """
        with self.lock:
            if self.latest_frame is None:
                return -1, None, 0.0
            return self.latest_index, self.latest_frame, self.latest_time

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def get_latest_pair(cam_left: LatestFrameCamera, cam_right: LatestFrameCamera):
    li, left, lt = cam_left.get_latest()
    ri, right, rt = cam_right.get_latest()
    if left is None or right is None:
        return False, None, None
    return True, left, right


# =============================================================================
# 2. 直接生成 1920x1080 view，不生成完整宽幅图
# =============================================================================

def stitch_view_roi_remap(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    maps_fast,
    params,
    crop_x: int,
    crop_y: int,
    view_w: int,
    view_h: int,
    seam_x: int,
    blend_width: int,
    right_x_shift: int,
    right_y_shift: int,
) -> np.ndarray:
    """
    从虚拟宽幅坐标系中直接裁出 view_w x view_h 画面。

    重点：
        不生成完整 wide = 3406x1201。
        只反推当前 1920x1080 视口覆盖到了哪些区域，然后只 remap 这些 ROI。

    虚拟宽幅结构：
        [left_keep][overlap][right_keep]

    这样做是你要 20 FPS 以上的关键。
    """
    if (left_raw.shape[1], left_raw.shape[0]) != maps_fast.raw_image_size:
        left_raw = cv2.resize(left_raw, maps_fast.raw_image_size)

    if (right_raw.shape[1], right_raw.shape[0]) != maps_fast.raw_image_size:
        right_raw = cv2.resize(right_raw, maps_fast.raw_image_size)

    output_w = int(params.output_width)
    output_h = int(params.output_height)

    crop_x = int(np.clip(crop_x, 0, max(0, output_w - view_w)))
    crop_y = int(np.clip(crop_y, 0, max(0, output_h - view_h)))

    view = np.zeros((view_h, view_w, 3), dtype=np.uint8)

    valid_h = min(view_h, output_h - crop_y)
    if valid_h <= 0:
        return view

    # 左右图在 rectified 坐标系中的 y 起点。
    left_y1 = params.left_y1 + crop_y
    left_y2 = left_y1 + valid_h
    right_y1 = params.right_y1 + crop_y + right_y_shift
    right_y2 = right_y1 + valid_h

    left_keep_w = params.left_keep_x2 - params.left_keep_x1
    overlap_w = params.overlap_px
    right_keep_w = params.right_keep_x2 - params.right_keep_x1

    left_keep_start = 0
    left_keep_end = left_keep_w
    overlap_start = left_keep_end
    overlap_end = overlap_start + overlap_w
    right_keep_start = overlap_end
    right_keep_end = right_keep_start + right_keep_w

    view_start = crop_x
    view_end = crop_x + view_w

    def paste_direct(seg_start, seg_end, raw, map1, map2, rect_x_base, rect_y1, rect_y2):
        """把普通非融合区间 remap 后贴到 view。"""
        ix1 = max(view_start, seg_start)
        ix2 = min(view_end, seg_end)
        if ix2 <= ix1:
            return

        dst_x1 = ix1 - view_start
        dst_x2 = ix2 - view_start

        rect_x1 = rect_x_base + (ix1 - seg_start)
        rect_x2 = rect_x_base + (ix2 - seg_start)

        roi = base.remap_rectified_roi_fixed_size(
            raw,
            map1,
            map2,
            rect_x1,
            rect_y1,
            rect_x2,
            rect_y2,
        )
        if roi is not None and roi.size > 0:
            view[:valid_h, dst_x1:dst_x2] = roi

    # 1. 左侧非重叠区域
    paste_direct(
        left_keep_start,
        left_keep_end,
        left_raw,
        maps_fast.left_map1,
        maps_fast.left_map2,
        params.left_keep_x1,
        left_y1,
        left_y2,
    )

    # 2. overlap 区域，根据 seam_x / blend_width 决定左右图边界
    ix1 = max(view_start, overlap_start)
    ix2 = min(view_end, overlap_end)

    if ix2 > ix1:
        ox1 = ix1 - overlap_start
        ox2 = ix2 - overlap_start

        blend_width = max(1, min(int(blend_width), overlap_w))
        if seam_x < 0:
            seam_x_used = overlap_w // 2
        else:
            seam_x_used = int(seam_x)

        half_blend = blend_width // 2
        seam_x_used = int(np.clip(
            seam_x_used,
            half_blend,
            overlap_w - (blend_width - half_blend),
        ))

        blend_x1 = seam_x_used - half_blend
        blend_x2 = blend_x1 + blend_width

        def paste_overlap_left(local_x1, local_x2):
            if local_x2 <= local_x1:
                return
            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start
            roi = base.remap_rectified_roi_fixed_size(
                left_raw,
                maps_fast.left_map1,
                maps_fast.left_map2,
                params.left_overlap_x1 + local_x1,
                left_y1,
                params.left_overlap_x1 + local_x2,
                left_y2,
            )
            if roi is not None and roi.size > 0:
                view[:valid_h, dst_x1:dst_x2] = roi

        def paste_overlap_right(local_x1, local_x2):
            if local_x2 <= local_x1:
                return
            dst_x1 = overlap_start + local_x1 - view_start
            dst_x2 = overlap_start + local_x2 - view_start
            roi = base.remap_rectified_roi_fixed_size(
                right_raw,
                maps_fast.right_map1,
                maps_fast.right_map2,
                params.right_overlap_x1 + local_x1 + right_x_shift,
                right_y1,
                params.right_overlap_x1 + local_x2 + right_x_shift,
                right_y2,
            )
            if roi is not None and roi.size > 0:
                view[:valid_h, dst_x1:dst_x2] = roi

        # blend 左侧：只用左图
        paste_overlap_left(ox1, min(ox2, blend_x1))

        # blend 右侧：只用右图
        paste_overlap_right(max(ox1, blend_x2), ox2)

        # seam 附近：做小范围融合
        bx1 = max(ox1, blend_x1)
        bx2 = min(ox2, blend_x2)

        if bx2 > bx1:
            dst_x1 = overlap_start + bx1 - view_start
            dst_x2 = overlap_start + bx2 - view_start

            left_part = base.remap_rectified_roi_fixed_size(
                left_raw,
                maps_fast.left_map1,
                maps_fast.left_map2,
                params.left_overlap_x1 + bx1,
                left_y1,
                params.left_overlap_x1 + bx2,
                left_y2,
            )
            right_part = base.remap_rectified_roi_fixed_size(
                right_raw,
                maps_fast.right_map1,
                maps_fast.right_map2,
                params.right_overlap_x1 + bx1 + right_x_shift,
                right_y1,
                params.right_overlap_x1 + bx2 + right_x_shift,
                right_y2,
            )

            if left_part is not None and right_part is not None and left_part.size > 0 and right_part.size > 0:
                alpha_line = np.linspace(0.0, 1.0, blend_width, dtype=np.float32)
                alpha = alpha_line[bx1 - blend_x1:bx2 - blend_x1].reshape(1, -1, 1)
                blended = left_part.astype(np.float32) * (1.0 - alpha) + right_part.astype(np.float32) * alpha
                view[:valid_h, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # 3. 右侧非重叠区域
    paste_direct(
        right_keep_start,
        right_keep_end,
        right_raw,
        maps_fast.right_map1,
        maps_fast.right_map2,
        params.right_keep_x1 + right_x_shift,
        right_y1,
        right_y2,
    )

    return view


# =============================================================================
# 3. 检测结果格式转换和画框
# =============================================================================

def wide_detections_to_boxes_info(wide_detections) -> List[dict]:
    """
    把 WideDetection 转成你的 predict1_weighted.py 使用的 boxes_info 格式。

    predict1_weighted.py 里的 get_person_boxes() 需要字段：
        cls_name, conf, x1, y1, x2, y2
    """
    boxes = []
    for det in wide_detections:
        x1, y1, x2, y2 = det.wide_bbox
        boxes.append({
            "cls_name": "person",
            "conf": float(det.score),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "source": det.source,
            "track_id": int(det.track_id),
        })
    return boxes



def choose_view_crop_from_director(
    director_state: dict,
    analysis: dict,
    wide_w: int,
    wide_h: int,
    view_w: int,
    view_h: int,
    crop_y_mode: str = "center",
    crop_state: Optional[dict] = None,
    crop_mode: str = "fixed",
    window_source: str = "current",
    window_crop_alpha: float = 0.12,
    window_crop_max_step: float = 24.0,
    window_crop_snap_px: float = 2.0,
) -> Tuple[int, int]:
    """
    根据导播状态生成最终 1920x1080 视口 crop_x / crop_y。

    这版恢复“三段固定窗口”逻辑：

        Left   -> crop_x = 0
        Center -> crop_x = (wide_w - view_w) / 2
        Right  -> crop_x = wide_w - view_w

    为什么要这样改：
        原来的快版只使用 director_state["current_anchor_x"]。
        这样虽然能动，但不一定真正贴到左/中/右三段边界。
        你观察到“最左边停留时没有达到缩略图最左边”，就是这个原因。

    crop_mode:
        fixed:
            使用 current_window / target_window 决定固定三段位置。
            这是推荐模式。

        anchor:
            使用原来的 current_anchor_x 连续锚点模式。
            仅用于对比调试。

    window_source:
        current:
            使用已经稳定生效的 current_window。
            这样窗口切换更稳，不会因为 target_window 抖动而来回移动。

        target:
            使用目标窗口 target_window。
            更跟手，但也更容易抖。
    """
    max_crop_x = max(0, int(wide_w - view_w))

    # ------------------------------------------------------------
    # 1. 横向 crop_x
    # ------------------------------------------------------------
    if crop_mode == "anchor":
        # 原来的连续锚点模式。
        anchor_x = director_state.get("current_anchor_x", wide_w * 0.5)
        target_crop_x = int(round(anchor_x - view_w * 0.5))
        target_crop_x = int(np.clip(target_crop_x, 0, max_crop_x))
        window_name = str(director_state.get("current_window", "Center"))
    else:
        # 固定三段窗口模式。
        if window_source == "target":
            window_name = str(director_state.get("target_window", "Center"))
        else:
            window_name = str(director_state.get("current_window", "Center"))

        wname = window_name.lower()

        if "left" in wname:
            target_crop_x = 0
        elif "right" in wname:
            target_crop_x = max_crop_x
        else:
            target_crop_x = max_crop_x // 2

    # ------------------------------------------------------------
    # 2. 对 crop_x 做平滑，避免 Left/Center/Right 切换时太快
    # ------------------------------------------------------------
    if crop_state is None:
        crop_x = target_crop_x
    else:
        prev_crop_x = crop_state.get("crop_x", None)

        if prev_crop_x is None:
            crop_x = target_crop_x
        else:
            prev_crop_x = float(prev_crop_x)
            delta = float(target_crop_x) - prev_crop_x

            if abs(delta) <= float(window_crop_snap_px):
                crop_x = float(target_crop_x)
            else:
                # EMA 平滑 + 每帧最大步长限制。
                step = delta * float(window_crop_alpha)

                max_step = max(1.0, float(window_crop_max_step))
                step = float(np.clip(step, -max_step, max_step))

                crop_x = prev_crop_x + step

        crop_x = int(round(np.clip(crop_x, 0, max_crop_x)))

        crop_state["crop_x"] = crop_x
        crop_state["target_crop_x"] = int(target_crop_x)
        crop_state["window_name"] = window_name
        crop_state["crop_mode"] = crop_mode

    # ------------------------------------------------------------
    # 3. 纵向 crop_y
    # ------------------------------------------------------------
    if crop_y_mode == "bottom":
        crop_y = max(0, wide_h - view_h)
    elif crop_y_mode == "focus":
        focus_y = analysis.get("focus_y", wide_h * VERTICAL_HOME_Y_RATIO)
        crop_y = int(round(focus_y - view_h * 0.5))
        crop_y = int(np.clip(crop_y, 0, max(0, wide_h - view_h)))
    else:
        crop_y = max(0, (wide_h - view_h) // 2)

    return int(crop_x), int(crop_y)


def draw_detections_on_view(
    view: np.ndarray,
    wide_detections,
    crop_x: int,
    crop_y: int,
    fps: float,
    last_infer_ms: float,
    total_ms: float,
    detect_side: str,
    frame_idx: int,
):
    """把宽幅坐标检测框转换到当前 1920x1080 view 坐标并绘制。"""
    out = view.copy()
    h, w = out.shape[:2]

    for det in wide_detections:
        x1, y1, x2, y2 = det.wide_bbox
        bx, by = det.wide_bottom

        vx1 = int(round(x1 - crop_x))
        vy1 = int(round(y1 - crop_y))
        vx2 = int(round(x2 - crop_x))
        vy2 = int(round(y2 - crop_y))
        vbx = int(round(bx - crop_x))
        vby = int(round(by - crop_y))

        # 完全不在 view 内的框跳过
        if vx2 < 0 or vx1 >= w or vy2 < 0 or vy1 >= h:
            continue

        vx1 = int(np.clip(vx1, 0, w - 1))
        vx2 = int(np.clip(vx2, 0, w - 1))
        vy1 = int(np.clip(vy1, 0, h - 1))
        vy2 = int(np.clip(vy2, 0, h - 1))

        color = (0, 255, 0) if det.source == "left" else (0, 180, 255)
        cv2.rectangle(out, (vx1, vy1), (vx2, vy2), color, 2)

        if 0 <= vbx < w and 0 <= vby < h:
            cv2.circle(out, (vbx, vby), 5, (0, 0, 255), -1)

        label = f"{det.source[0].upper()} id={det.track_id} {det.score:.2f}"
        cv2.putText(out, label, (vx1, max(20, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    text = (
        f"FPS:{fps:.1f} infer:{last_infer_ms:.1f}ms total:{total_ms:.1f}ms "
        f"detect:{detect_side} crop=({crop_x},{crop_y}) persons:{len(wide_detections)}"
    )
    cv2.putText(out, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(out, f"frame:{frame_idx}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out




def draw_wide_minimap_on_view(
    view: np.ndarray,
    wide_w: int,
    wide_h: int,
    crop_x: int,
    crop_y: int,
    view_w: int,
    view_h: int,
    wide_detections,
    params=None,
    minimap_width: int = 420,
    margin: int = 18,
) -> np.ndarray:
    """
    在 1920x1080 输出画面右下角绘制“完整宽图缩略示意图”。

    这个缩略图不是实时生成真实宽幅图，而是一个坐标示意图：
        1. 灰色长条代表完整宽幅图。
        2. 白色矩形代表当前正在输出的 1920x1080 裁切窗口。
        3. 小圆点代表检测到的人在完整宽图中的位置。
        4. 竖线代表 left_keep / overlap / right_keep 的分界。

    为什么不直接生成真实宽图缩略图：
        真实宽图需要额外 remap 和 resize，会增加开销。
        当前目标是保证运镜输出 20 FPS 以上，所以这里使用轻量坐标示意图。
    """
    if view is None or view.size == 0:
        return view

    out = view.copy()
    h, w = out.shape[:2]

    wide_w = max(1, int(wide_w))
    wide_h = max(1, int(wide_h))
    view_w = max(1, int(view_w))
    view_h = max(1, int(view_h))

    # 缩略图宽度不能超过画面宽度。
    thumb_w = int(min(max(160, minimap_width), w - margin * 2))
    thumb_h = int(round(thumb_w * wide_h / wide_w))
    thumb_h = max(60, min(thumb_h, h // 3))

    x0 = w - thumb_w - margin
    y0 = h - thumb_h - margin

    # 背景框范围，稍微比缩略图大一点，方便看清文字。
    bg_x1 = max(0, x0 - 10)
    bg_y1 = max(0, y0 - 34)
    bg_x2 = min(w, x0 + thumb_w + 10)
    bg_y2 = min(h, y0 + thumb_h + 12)

    # 半透明黑底。
    overlay = out.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    out[bg_y1:bg_y2, bg_x1:bg_x2] = cv2.addWeighted(
        overlay[bg_y1:bg_y2, bg_x1:bg_x2],
        0.62,
        out[bg_y1:bg_y2, bg_x1:bg_x2],
        0.38,
        0,
    )

    # 缩略图主体。
    cv2.rectangle(out, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (45, 45, 45), -1)
    cv2.rectangle(out, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (220, 220, 220), 1)

    # 画 left_keep / overlap / right_keep 分界线。
    if params is not None:
        try:
            left_keep_w = int(params.left_keep_x2 - params.left_keep_x1)
            overlap_w = int(params.overlap_px)

            split1 = x0 + int(round(left_keep_w / wide_w * thumb_w))
            split2 = x0 + int(round((left_keep_w + overlap_w) / wide_w * thumb_w))

            cv2.line(out, (split1, y0), (split1, y0 + thumb_h), (80, 180, 255), 1)
            cv2.line(out, (split2, y0), (split2, y0 + thumb_h), (80, 180, 255), 1)

            cv2.putText(
                out,
                "L",
                (x0 + 5, y0 + thumb_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 220, 255),
                1,
            )
            cv2.putText(
                out,
                "O",
                (split1 + 5, y0 + thumb_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 220, 255),
                1,
            )
            cv2.putText(
                out,
                "R",
                (split2 + 5, y0 + thumb_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 220, 255),
                1,
            )
        except Exception:
            pass

    # 当前 1920x1080 裁切框在完整宽图中的位置。
    rx1 = x0 + int(round(crop_x / wide_w * thumb_w))
    ry1 = y0 + int(round(crop_y / wide_h * thumb_h))
    rx2 = x0 + int(round((crop_x + view_w) / wide_w * thumb_w))
    ry2 = y0 + int(round((crop_y + view_h) / wide_h * thumb_h))

    rx1 = int(np.clip(rx1, x0, x0 + thumb_w))
    rx2 = int(np.clip(rx2, x0, x0 + thumb_w))
    ry1 = int(np.clip(ry1, y0, y0 + thumb_h))
    ry2 = int(np.clip(ry2, y0, y0 + thumb_h))

    # 当前视口框：白色外框 + 黄色内框，更醒目。
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
    cv2.rectangle(out, (rx1 + 2, ry1 + 2), (max(rx1 + 3, rx2 - 2), max(ry1 + 3, ry2 - 2)), (0, 255, 255), 1)

    # 人物点位。
    for det in wide_detections:
        try:
            bx, by = det.wide_bottom
            px = x0 + int(round(bx / wide_w * thumb_w))
            py = y0 + int(round(by / wide_h * thumb_h))

            if x0 <= px <= x0 + thumb_w and y0 <= py <= y0 + thumb_h:
                color = (0, 255, 0) if det.source == "left" else (0, 180, 255)
                cv2.circle(out, (px, py), 4, color, -1)
                cv2.circle(out, (px, py), 5, (0, 0, 0), 1)
        except Exception:
            continue

    # 标题和当前 crop 坐标。
    cv2.putText(
        out,
        f"WIDE MAP  crop=({int(crop_x)},{int(crop_y)})",
        (x0, max(18, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    return out


# =============================================================================
# 4. 主流程
# =============================================================================

def read_terminal_key() -> int:
    """支持终端输入 q/s/l 回车控制。"""
    try:
        if not sys.stdin.isatty():
            return 255
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return 255
        line = sys.stdin.readline().strip()
        if not line:
            return 255
        return ord(line[0])
    except Exception:
        return 255


def run(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    base.ensure_dir(args.save_dir)

    # 1. 加载 map / stitch 参数。
    maps_float, maps_fast = base.load_rectify_maps(args.map_file)
    params = base.load_stitch_params(args.stitch_param)

    wide_w = int(params.output_width)
    wide_h = int(params.output_height)
    out_ratio = args.view_width / max(1.0, args.view_height)

    # 2. 加载或构造 raw -> rectified 反查表。
    inv_maps = base.load_or_build_inverse_maps(
        maps_float,
        map_file=args.map_file,
        cache_file=args.inverse_cache,
        rebuild=args.rebuild_inverse,
    )

    # 3. 左路 left_keep mask：左路只检测非重叠区域。
    left_keep_mask = base.build_left_keep_raw_mask(
        maps_float,
        params,
        dilate_iter=args.left_mask_dilate,
    )
    mask_left_overlap = not args.disable_left_keep_mask
    print(f"[信息] 左路 overlap 屏蔽: {mask_left_overlap}")

    # 4. RKNN 检测器：单实例，左右交替检测。
    detector = base.PersonDetector(
        model_path=args.model,
        labels_path=args.labels,
        obj_thresh=args.conf,
        nms_thresh=args.nms,
        input_size=args.input_size,
        core_id=args.rknn_core,
        use_rgb=not args.bgr_input,
        name="director-rknn",
    )

    # 5. 后台采集左右摄像头。
    cam_left = LatestFrameCamera(args.left_device, args.width, args.height, args.fps, name="left").start()
    cam_right = LatestFrameCamera(args.right_device, args.width, args.height, args.fps, name="right").start()

    print("[信息] 等待摄像头第一帧...")
    if not cam_left.wait_first_frame(timeout=3.0):
        raise RuntimeError("左摄像头 3 秒内没有读到第一帧")
    if not cam_right.wait_first_frame(timeout=3.0):
        raise RuntimeError("右摄像头 3 秒内没有读到第一帧")
    print("[信息] 后台采集线程已启动")

    # 6. 检测结果缓存和平滑器。
    last_left_results = []
    last_right_results = []
    last_left_frame_idx = -1
    last_right_frame_idx = -1
    detect_turn = 0
    last_detect_side = "none"
    last_infer_ms = 0.0

    smoother = base.SmoothTracks(
        smooth=args.smooth,
        max_match_dist=args.smooth_match_dist,
        max_missing=args.smooth_max_missing,
    )

    # 7. 你的运镜状态。
    analysis_state = init_single_view_state()
    director_state = init_overlay_director_state(wide_w)

    # 三段固定窗口裁切状态。
    # 用于保存上一帧 crop_x，使 Left/Center/Right 切换时平滑移动。
    crop_state = {
        "crop_x": None,
        "target_crop_x": None,
        "window_name": "Center",
    }

    fps_counter = base.FPSCounter()

    window_name = "RKNN director view 1920x1080"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    debug_idx = 0

    print("\n[信息] 开始运行：最终输出 1920x1080 view。")
    print("按键：q/ESC 退出，s 保存当前 view，l 切换左路 overlap 屏蔽。")
    print("建议：如果目标是 20 FPS 以上，优先使用 --detect-interval 3 或 4。\n")

    try:
        while not STOP_REQUESTED:
            loop_t0 = time.perf_counter()

            ok, left_raw, right_raw = get_latest_pair(cam_left, cam_right)
            if not ok:
                time.sleep(0.002)
                continue

            # ------------------------------------------------------------
            # A. 低频 RKNN 检测：每隔 detect_interval 帧只检测一路
            # ------------------------------------------------------------
            do_detect = (frame_idx % max(1, args.detect_interval) == 0)

            if do_detect:
                if detect_turn % 2 == 0:
                    detect_input = left_raw
                    if mask_left_overlap:
                        detect_input = base.apply_detection_mask(left_raw, left_keep_mask)

                    t0 = time.perf_counter()
                    last_left_results = detector.detect(detect_input)
                    t1 = time.perf_counter()

                    last_left_frame_idx = frame_idx
                    last_detect_side = "left"
                    last_infer_ms = (t1 - t0) * 1000.0
                else:
                    t0 = time.perf_counter()
                    last_right_results = detector.detect(right_raw)
                    t1 = time.perf_counter()

                    last_right_frame_idx = frame_idx
                    last_detect_side = "right"
                    last_infer_ms = (t1 - t0) * 1000.0

                detect_turn += 1

            # ------------------------------------------------------------
            # B. raw 检测结果 -> wide 坐标
            # ------------------------------------------------------------
            left_wide = base.convert_results_to_wide(
                source="left",
                raw_results=last_left_results,
                inv_maps=inv_maps,
                params=params,
                right_x_shift=args.runtime_right_x_shift,
                right_y_shift=args.runtime_right_y_shift,
                search_radius=args.inverse_search_radius,
                allow_left_overlap=False,
            )

            right_wide = base.convert_results_to_wide(
                source="right",
                raw_results=last_right_results,
                inv_maps=inv_maps,
                params=params,
                right_x_shift=args.runtime_right_x_shift,
                right_y_shift=args.runtime_right_y_shift,
                search_radius=args.inverse_search_radius,
                allow_left_overlap=True,
            )

            wide_detections = smoother.update(left_wide + right_wide)

            # ------------------------------------------------------------
            # C. 运镜算法：每帧用最近检测结果更新导播状态
            # ------------------------------------------------------------
            director_t0 = time.perf_counter()
            boxes_info = wide_detections_to_boxes_info(wide_detections)
            analysis = analyze_single_view_frame(boxes_info, wide_w, wide_h, analysis_state)
            director_info = update_overlay_director_state(
                analysis,
                wide_w,
                wide_h,
                out_ratio,
                director_state,
            )
            crop_x, crop_y = choose_view_crop_from_director(
                director_state,
                analysis,
                wide_w,
                wide_h,
                args.view_width,
                args.view_height,
                crop_y_mode=args.crop_y_mode,
                crop_state=crop_state,
                crop_mode=args.window_crop_mode,
                window_source=args.window_source,
                window_crop_alpha=args.window_crop_alpha,
                window_crop_max_step=args.window_crop_max_step,
                window_crop_snap_px=args.window_crop_snap_px,
            )
            director_t1 = time.perf_counter()

            # ------------------------------------------------------------
            # D. 只生成 1920x1080 view，不生成完整 wide
            # ------------------------------------------------------------
            view_t0 = time.perf_counter()
            view = stitch_view_roi_remap(
                left_raw=left_raw,
                right_raw=right_raw,
                maps_fast=maps_fast,
                params=params,
                crop_x=crop_x,
                crop_y=crop_y,
                view_w=args.view_width,
                view_h=args.view_height,
                seam_x=args.runtime_seam_x,
                blend_width=args.runtime_blend_width,
                right_x_shift=args.runtime_right_x_shift,
                right_y_shift=args.runtime_right_y_shift,
            )
            view_t1 = time.perf_counter()

            loop_t1 = time.perf_counter()
            total_ms = (loop_t1 - loop_t0) * 1000.0
            director_ms = (director_t1 - director_t0) * 1000.0
            view_ms = (view_t1 - view_t0) * 1000.0
            fps = fps_counter.update()

            left_age = frame_idx - last_left_frame_idx if last_left_frame_idx >= 0 else -1
            right_age = frame_idx - last_right_frame_idx if last_right_frame_idx >= 0 else -1

            vis = draw_detections_on_view(
                view,
                wide_detections,
                crop_x,
                crop_y,
                fps=fps,
                last_infer_ms=last_infer_ms,
                total_ms=total_ms,
                detect_side=last_detect_side,
                frame_idx=frame_idx,
            )

            # 右下角宽图缩略示意图：
            # 用很小的绘图开销显示“当前 1920x1080 裁切窗口在完整宽图中的位置”。
            if not args.no_minimap:
                vis = draw_wide_minimap_on_view(
                    vis,
                    wide_w=wide_w,
                    wide_h=wide_h,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    view_w=args.view_width,
                    view_h=args.view_height,
                    wide_detections=wide_detections,
                    params=params,
                    minimap_width=args.minimap_width,
                )

            # ------------------------------------------------------------
            # E. 显示 / 按键
            # ------------------------------------------------------------
            if not args.headless:
                show = base.resize_for_display(vis, args.display_scale)
                cv2.imshow(window_name, show)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            term_key = read_terminal_key()
            if term_key != 255:
                key = term_key

            if key in (ord("q"), 27):
                print("[信息] 用户退出")
                break
            elif key == ord("s"):
                path = os.path.join(args.save_dir, f"view1920_{debug_idx:04d}_{base.current_timestamp()}.jpg")
                cv2.imwrite(path, vis)
                print(f"[信息] 已保存: {path}")
                debug_idx += 1
            elif key == ord("l"):
                mask_left_overlap = not mask_left_overlap
                print(f"[信息] 左路 overlap 屏蔽: {mask_left_overlap}")

            # ------------------------------------------------------------
            # F. 周期性输出性能
            # ------------------------------------------------------------
            if args.print_every > 0 and frame_idx % args.print_every == 0:
                print(
                    f"[PROFILE] frame={frame_idx} fps={fps:.1f} "
                    f"detect={last_detect_side} infer={last_infer_ms:.1f}ms "
                    f"director={director_ms:.1f}ms view={view_ms:.1f}ms total={total_ms:.1f}ms "
                    f"L={len(last_left_results)} age={left_age} "
                    f"R={len(last_right_results)} age={right_age} "
                    f"persons={len(wide_detections)} "
                    f"window={crop_state.get('window_name')} "
                    f"target_crop={crop_state.get('target_crop_x')} "
                    f"crop=({crop_x},{crop_y})"
                )

            frame_idx += 1

    finally:
        try:
            cam_left.stop()
            cam_right.stop()
        except Exception:
            pass

        try:
            detector.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[信息] 程序退出")


# =============================================================================
# 5. 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双路 RKNN + 运镜 + 直接输出 1920x1080 view")

    parser.add_argument("--left-device", default=base.DEFAULT_LEFT_DEVICE)
    parser.add_argument("--right-device", default=base.DEFAULT_RIGHT_DEVICE)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--model", default=base.DEFAULT_MODEL)
    parser.add_argument("--labels", default=base.DEFAULT_LABELS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--rknn-core", type=int, default=-1, help="-1=全部 NPU core；0/1/2=指定单核")
    parser.add_argument("--bgr-input", action="store_true", help="如果模型输入是 BGR，则打开此项；默认 RGB")

    parser.add_argument("--map-file", default=base.DEFAULT_MAP_FILE)
    parser.add_argument("--stitch-param", default=base.DEFAULT_STITCH_PARAM)
    parser.add_argument("--inverse-cache", default=None)
    parser.add_argument("--rebuild-inverse", action="store_true")
    parser.add_argument("--inverse-search-radius", type=int, default=8)

    parser.add_argument("--runtime-seam-x", type=int, default=150)
    parser.add_argument("--runtime-blend-width", type=int, default=40)
    parser.add_argument("--runtime-right-x-shift", type=int, default=30)
    parser.add_argument("--runtime-right-y-shift", type=int, default=-5)

    parser.add_argument("--view-width", type=int, default=1920)
    parser.add_argument("--view-height", type=int, default=1080)
    parser.add_argument(
        "--crop-y-mode",
        choices=["center", "bottom", "focus"],
        default="center",
        help="纵向裁剪策略。center 最稳；bottom 更偏下；focus 跟随人物纵向焦点",
    )

    parser.add_argument(
        "--window-crop-mode",
        choices=["fixed", "anchor"],
        default="fixed",
        help="横向运镜模式。fixed=固定 Left/Center/Right 三段；anchor=原来的连续锚点模式",
    )
    parser.add_argument(
        "--window-source",
        choices=["current", "target"],
        default="current",
        help="fixed 模式下使用 current_window 还是 target_window。current 更稳，target 更跟手",
    )
    parser.add_argument(
        "--window-crop-alpha",
        type=float,
        default=0.12,
        help="三段窗口 crop_x 平滑系数，越大移动越快",
    )
    parser.add_argument(
        "--window-crop-max-step",
        type=float,
        default=24.0,
        help="三段窗口每帧最大横向移动像素。调小更慢更稳，调大更快",
    )
    parser.add_argument(
        "--window-crop-snap-px",
        type=float,
        default=2.0,
        help="距离目标小于该像素时直接吸附到目标，确保最终能到达最左/最右",
    )

    parser.add_argument(
        "--detect-interval",
        type=int,
        default=3,
        help="每隔 N 帧做一次 RKNN 检测。目标 20FPS 建议 3 或 4",
    )
    parser.add_argument("--smooth", type=float, default=0.70)
    parser.add_argument("--smooth-match-dist", type=float, default=180.0)
    parser.add_argument("--smooth-max-missing", type=int, default=20)

    parser.add_argument("--disable-left-keep-mask", action="store_true")
    parser.add_argument("--left-mask-dilate", type=int, default=3)

    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--no-minimap", action="store_true", help="关闭右下角完整宽图缩略示意图")
    parser.add_argument("--minimap-width", type=int, default=420, help="右下角宽图缩略示意图宽度，建议 360~520")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-dir", default="/home/elf/work/basketball/camera_movement/debug_view1920")
    parser.add_argument("--print-every", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ RKNN 运镜直接输出 1920x1080 ================")
    print(f"left_device        : {args.left_device}")
    print(f"right_device       : {args.right_device}")
    print(f"camera size        : {args.width} x {args.height}")
    print(f"model              : {args.model}")
    print(f"map_file           : {args.map_file}")
    print(f"stitch_param       : {args.stitch_param}")
    print(f"view size          : {args.view_width} x {args.view_height}")
    print(f"detect_interval    : {args.detect_interval}")
    print(f"smooth             : {args.smooth}")
    print(f"runtime seam/blend : {args.runtime_seam_x} / {args.runtime_blend_width}")
    print(f"right shift        : x={args.runtime_right_x_shift}, y={args.runtime_right_y_shift}")
    print("===========================================================\n")

    run(args)


if __name__ == "__main__":
    main()
