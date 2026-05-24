import cv2
import numpy as np

from predict1_weighted import *
from predict1_yolo import draw_boxes_on_frame

def choose_window_by_pan_bias(pan_bias, current_window):
    if current_window == WINDOW_RIGHT:
        if pan_bias >= WINDOW_HOLD_BIAS:
            return WINDOW_RIGHT
        if pan_bias <= -WINDOW_ENTER_BIAS:
            return WINDOW_LEFT
        return WINDOW_CENTER

    if current_window == WINDOW_LEFT:
        if pan_bias <= -WINDOW_HOLD_BIAS:
            return WINDOW_LEFT
        if pan_bias >= WINDOW_ENTER_BIAS:
            return WINDOW_RIGHT
        return WINDOW_CENTER

    if pan_bias >= WINDOW_ENTER_BIAS:
        return WINDOW_RIGHT
    if pan_bias <= -WINDOW_ENTER_BIAS:
        return WINDOW_LEFT
    return WINDOW_CENTER


def get_virtual_binocular_geometry(img_w):
    side_pad = img_w * VIRTUAL_BINOCULAR_SIDE_PAD_RATIO
    virtual_w = img_w + side_pad * 2.0
    return virtual_w, side_pad


def to_virtual_x(x, img_w):
    virtual_w, side_pad = get_virtual_binocular_geometry(img_w)
    return x + side_pad, virtual_w, side_pad


def get_window_box_norm(window_name):
    if window_name == WINDOW_LEFT:
        return WINDOW_LEFT_BOX
    if window_name == WINDOW_RIGHT:
        return WINDOW_RIGHT_BOX
    return WINDOW_CENTER_BOX


def interpolate_window_box_norm(from_window, to_window, progress):
    from_box = get_window_box_norm(from_window)
    to_box = get_window_box_norm(to_window)
    p = smoothstep01(progress)
    return tuple(from_box[i] * (1.0 - p) + to_box[i] * p for i in range(4))


def focus_in_window_box(window_name, focus_x, focus_y, img_w, img_h):
    if focus_x is None or focus_y is None:
        return False

    norm_x = focus_x / max(1.0, img_w)
    norm_y = focus_y / max(1.0, img_h)
    box_x1, box_y1, box_x2, box_y2 = get_window_box_norm(window_name)
    return box_x1 <= norm_x <= box_x2 and box_y1 <= norm_y <= box_y2


def focus_window_distance(window_name, focus_x, focus_y, img_w, img_h):
    norm_x = focus_x / max(1.0, img_w)
    norm_y = focus_y / max(1.0, img_h)

    box_x1, box_y1, box_x2, box_y2 = get_window_box_norm(window_name)
    center_x = (box_x1 + box_x2) * 0.50
    center_y = (box_y1 + box_y2) * 0.50
    span_x = max(1e-6, box_x2 - box_x1)
    span_y = max(1e-6, box_y2 - box_y1)

    dx = (norm_x - center_x) / span_x
    dy = (norm_y - center_y) / span_y
    return dx * dx + dy * dy


def window_density_score(window_name, persons, focus_x, focus_y, img_w, img_h):
    box_x1, box_y1, box_x2, box_y2 = get_window_box_norm(window_name)
    span_x = max(1e-6, box_x2 - box_x1)
    span_y = max(1e-6, box_y2 - box_y1)
    score = 0.0

    for person in persons:
        cx, cy = box_center(person)
        norm_x = cx / max(1.0, img_w)
        norm_y = cy / max(1.0, img_h)
        weight = person_focus_weight(person, img_w, img_h)

        in_x = box_x1 <= norm_x <= box_x2
        in_y = box_y1 <= norm_y <= box_y2
        if in_x and in_y:
            score += weight
            continue

        dx = 0.0 if in_x else min(abs(norm_x - box_x1), abs(norm_x - box_x2)) / span_x
        dy = 0.0 if in_y else min(abs(norm_y - box_y1), abs(norm_y - box_y2)) / span_y
        proximity = clamp(1.0 - 0.85 * dx - 0.55 * dy, 0.0, 1.0)
        score += weight * proximity * 0.28

    if focus_x is not None and focus_y is not None:
        score += 1.35 * clamp(1.0 - math.sqrt(focus_window_distance(window_name, focus_x, focus_y, img_w, img_h)), 0.0, 1.0)

    return score


def choose_window_by_group_density(persons, focus_x, focus_y, img_w, img_h, current_window):
    if not persons:
        if focus_x is None or focus_y is None:
            return current_window
        return min(
            (WINDOW_LEFT, WINDOW_CENTER, WINDOW_RIGHT),
            key=lambda window_name: focus_window_distance(window_name, focus_x, focus_y, img_w, img_h)
        )

    scored_windows = []
    for window_name in (WINDOW_LEFT, WINDOW_CENTER, WINDOW_RIGHT):
        score = window_density_score(window_name, persons, focus_x, focus_y, img_w, img_h)
        if window_name == current_window:
            score += WINDOW_SCORE_HOLD_BONUS
        scored_windows.append((score, window_name))

    scored_windows.sort(reverse=True, key=lambda item: item[0])
    best_score, best_window = scored_windows[0]
    second_score = scored_windows[1][0] if len(scored_windows) > 1 else -1.0

    if best_window != current_window and best_score < second_score * 1.06:
        return current_window

    return best_window


def required_window_stable_frames(current_window, target_window):
    if target_window == WINDOW_CENTER:
        return WINDOW_CENTER_RETURN_STABLE_FRAMES
    if current_window == WINDOW_CENTER:
        return WINDOW_SIDE_STABLE_FRAMES
    return WINDOW_SIDE_TO_SIDE_STABLE_FRAMES


def resolve_target_window(current_window, target_window):
    if current_window == WINDOW_LEFT and target_window == WINDOW_RIGHT:
        return WINDOW_CENTER
    if current_window == WINDOW_RIGHT and target_window == WINDOW_LEFT:
        return WINDOW_CENTER
    return target_window


def update_window_with_hysteresis(
    current_window,
    target_window,
    pending_window,
    stable_count,
    dwell_frames,
    switch_cooldown
):
    target_window = resolve_target_window(current_window, target_window)

    if switch_cooldown > 0:
        switch_cooldown -= 1

    if target_window == current_window:
        return current_window, target_window, 0, dwell_frames + 1, False, switch_cooldown

    if target_window != WINDOW_CENTER and dwell_frames < WINDOW_MIN_DWELL_FRAMES:
        return current_window, target_window, 0, dwell_frames + 1, False, switch_cooldown

    if switch_cooldown > 0 and target_window != WINDOW_CENTER:
        return current_window, target_window, 0, dwell_frames + 1, False, switch_cooldown

    if target_window != pending_window:
        pending_window = target_window
        stable_count = 1
    else:
        stable_count += 1

    need_frames = required_window_stable_frames(current_window, target_window)
    if stable_count >= need_frames:
        return target_window, target_window, 0, 0, True, WINDOW_SWITCH_COOLDOWN_FRAMES

    return current_window, pending_window, stable_count, dwell_frames + 1, False, switch_cooldown


def get_window_guidance_anchor(from_window, to_window, progress, img_w, crop_w):
    box_x1, _, box_x2, _ = interpolate_window_box_norm(from_window, to_window, progress)
    guide_center_x = img_w * ((box_x1 + box_x2) * 0.50)
    return clamp(guide_center_x, crop_w / 2.0, img_w - crop_w / 2.0)


def apply_window_guidance(raw_target_anchor_x, from_window, to_window, progress, img_w, crop_w, fast_break=False):
    guide_anchor_x = get_window_guidance_anchor(from_window, to_window, progress, img_w, crop_w)
    blend = WINDOW_FAST_BREAK_GUIDE_BLEND if fast_break else WINDOW_GUIDE_BLEND
    if to_window == WINDOW_CENTER:
        blend *= 1.10
    return raw_target_anchor_x * (1.0 - blend) + guide_anchor_x * blend


def update_move_progress_display(
    display_from_window,
    display_to_window,
    display_progress,
    previous_window,
    current_window,
    switched
):
    active_transition = display_from_window != display_to_window

    if switched and previous_window != current_window:
        if (
            active_transition
            and display_from_window == current_window
            and display_to_window == previous_window
        ):
            display_from_window = previous_window
            display_to_window = current_window
            display_progress = 1.0 - display_progress
        else:
            display_from_window = previous_window
            display_to_window = current_window
            display_progress = 0.0
        active_transition = display_from_window != display_to_window

    if not active_transition:
        return current_window, current_window, 0.0, MOVE_STATIONARY

    if display_progress >= 0.999 and not switched:
        return current_window, current_window, 0.0, MOVE_STATIONARY

    display_progress = min(1.0, display_progress + MOVE_PROGRESS_MAX_STEP)
    progress_pct = int(round(display_progress * 100.0))
    return display_from_window, display_to_window, display_progress, f"{display_to_window} {progress_pct}%"


def resolve_move_target_window(current_window, pending_window, target_window):
    if pending_window != current_window:
        return pending_window
    if target_window != current_window:
        return target_window
    return current_window


def get_side_box_scale():
    return WINDOW_TRACK_BOX_SIDE_HEIGHT_RATIO / max(0.001, WINDOW_TRACK_BOX_HEIGHT_RATIO)


def interpolate_box_scale(from_scale, to_scale, progress):
    eased_progress = smoothstep01(progress)
    return from_scale * (1.0 - eased_progress) + to_scale * eased_progress


def scale_box_from_bottom_center(x1, y1, x2, y2, scale, out_w, out_h):
    scale = max(0.05, scale)
    base_w = max(2.0, x2 - x1)
    base_h = max(2.0, y2 - y1)
    anchor_x = (x1 + x2) * 0.50
    anchor_y = y2

    scaled_w = base_w * scale
    scaled_h = base_h * scale

    sx1 = anchor_x - scaled_w * 0.50
    sx2 = anchor_x + scaled_w * 0.50
    sy2 = anchor_y
    sy1 = anchor_y - scaled_h

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sx2 > out_w - 1:
        shift = sx2 - (out_w - 1)
        sx1 -= shift
        sx2 = out_w - 1
    sx1 = clamp(sx1, 0, max(0, sx2 - 1))
    sx2 = clamp(sx2, sx1 + 1, out_w - 1)

    sy2 = clamp(sy2, 1, out_h - 1)
    sy1 = clamp(sy1, 0, sy2 - 1)
    return sx1, sy1, sx2, sy2


def get_window_guide_box_rect(out_w, out_h, from_window, to_window, progress, box_scale=None):
    box_x1, _, box_x2, _ = interpolate_window_box_norm(from_window, to_window, progress)
    line_y = out_h * WINDOW_TRACK_LINE_Y_RATIO
    base_box_h = out_h * WINDOW_TRACK_BOX_HEIGHT_RATIO
    x1 = out_w * clamp(box_x1, 0.0, 1.0)
    x2 = out_w * clamp(box_x2, 0.0, 1.0)
    base_y1 = clamp(line_y, 0, out_h - 1)
    y2 = clamp(base_y1 + base_box_h, base_y1 + 1, out_h - 1)
    y1 = clamp(base_y1, 0, y2 - 1)
    scale = 1.0 if box_scale is None else box_scale
    return scale_box_from_bottom_center(x1, y1, x2, y2, scale, out_w, out_h)


def box_rect_to_pixels(box_rect, out_w, out_h):
    x1, y1, x2, y2 = box_rect
    ix1 = clamp(int(round(x1)), 0, out_w - 2)
    ix2 = clamp(int(round(x2)), ix1 + 1, out_w - 1)
    iy1 = clamp(int(round(y1)), 0, out_h - 2)
    iy2 = clamp(int(round(y2)), iy1 + 1, out_h - 1)
    return ix1, iy1, ix2, iy2


def smooth_box_rect(prev_rect, target_rect, alpha=BOX_DISPLAY_SMOOTH_ALPHA):
    if prev_rect is None:
        return target_rect
    return tuple(
        smooth_value(prev_value, target_value, alpha)
        for prev_value, target_value in zip(prev_rect, target_rect)
    )


def draw_window_guide_box(output_frame, from_window, to_window, progress, box_scale=None, box_rect=None):
    out_h, out_w = output_frame.shape[:2]
    line_y = int(round(out_h * WINDOW_TRACK_LINE_Y_RATIO))
    if box_rect is None:
        box_rect = get_window_guide_box_rect(
            out_w,
            out_h,
            from_window,
            to_window,
            progress,
            box_scale=box_scale,
        )
    x1, y1, x2, y2 = box_rect_to_pixels(box_rect, out_w, out_h)
    cv2.line(output_frame, (0, line_y), (out_w - 1, line_y), (80, 80, 255), 2)
    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return output_frame


def even_int(value):
    value = max(2, int(round(value)))
    return value if value % 2 == 0 else value + 1


def get_program_output_size(frame_w, frame_h):
    box_h = max(2, int(round(frame_h * WINDOW_TRACK_BOX_HEIGHT_RATIO)))
    widths = []
    for window_name in (WINDOW_LEFT, WINDOW_CENTER, WINDOW_RIGHT):
        box_x1, _, box_x2, _ = get_window_box_norm(window_name)
        widths.append(max(2, int(round(frame_w * (box_x2 - box_x1)))))

    return even_int(max(widths)), even_int(box_h)


def resize_to_canvas(frame, out_w, out_h):
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    scale = min(out_w / src_w, out_h / src_h)
    draw_w = max(1, int(round(src_w * scale)))
    draw_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (draw_w, draw_h))

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    off_x = max(0, (out_w - draw_w) // 2)
    off_y = max(0, (out_h - draw_h) // 2)
    canvas[off_y:off_y + draw_h, off_x:off_x + draw_w] = resized
    return canvas


def extract_program_frame(frame, from_window, to_window, progress, out_w, out_h, box_scale=None, box_rect=None):
    frame_h, frame_w = frame.shape[:2]
    if box_rect is None:
        box_rect = get_window_guide_box_rect(
            frame_w,
            frame_h,
            from_window,
            to_window,
            progress,
            box_scale=box_scale,
        )
    x1, y1, x2, y2 = box_rect_to_pixels(box_rect, frame_w, frame_h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return resize_to_canvas(frame, out_w, out_h)
    return resize_to_canvas(crop, out_w, out_h)

def get_crop_size(scale_value, img_w, img_h, out_ratio):
    crop_w = img_w / scale_value
    crop_h = crop_w / out_ratio

    if crop_h > img_h:
        crop_h = img_h / scale_value
        crop_w = crop_h * out_ratio

    return crop_w, crop_h


def focus_to_pan_bias(focus_x, img_w):
    if focus_x is None:
        return 0.0

    density_offset = (focus_x / max(1.0, img_w) - 0.50) * 2.0
    if abs(density_offset) <= 0.04:
        return 0.0

    return clamp(density_offset * PAN_DENSITY_GAIN, -1.0, 1.0)


def battle_hotspot_center(main_group, main_region, img_w, img_h, ball_x=None, attack_direction=0):
    density_x, _, density_confidence = density_cluster_focus(main_group, img_w, img_h)
    region_x, _ = region_center(main_region)
    weighted_x, _ = weighted_focus(main_group, img_w, img_h)

    xs = [x for x in (density_x, region_x, weighted_x) if x is not None]
    if not xs:
        return None, 0.0

    if density_x is None:
        density_x = sum(xs) / len(xs)
    if region_x is None:
        region_x = density_x
    if weighted_x is None:
        weighted_x = density_x

    hotspot_x = density_x * 0.84 + region_x * 0.10 + weighted_x * 0.06

    edge_x = attack_edge_x(main_group, img_w, img_h, direction=attack_direction)
    if edge_x is not None:
        hotspot_x = hotspot_x * (1.0 - HOTSPOT_ATTACK_EDGE_BLEND) + edge_x * HOTSPOT_ATTACK_EDGE_BLEND

    if ball_x is not None:
        hotspot_x = hotspot_x * (1.0 - HOTSPOT_BALL_BLEND) + ball_x * HOTSPOT_BALL_BLEND

    group_size_confidence = clamp(len(main_group) / 6.0, 0.35, 1.0)
    confidence = clamp(0.78 * density_confidence + 0.22 * group_size_confidence, 0.45, 1.0)
    if ball_x is not None:
        confidence = clamp(confidence + 0.08, 0.45, 1.0)

    return hotspot_x, confidence


def scale_pan_bias_by_group_size(pan_bias, group_size):
    if group_size >= PAN_SMALL_GROUP_FULL_SIZE:
        return pan_bias

    scale = clamp(
        group_size / max(1.0, PAN_SMALL_GROUP_FULL_SIZE),
        PAN_SMALL_GROUP_MIN_SCALE,
        1.0
    )
    return pan_bias * scale


def get_pan_bias_center(pan_bias, img_w, crop_w):
    pan_range = max(0.0, img_w - crop_w)
    return img_w * 0.50 + clamp(pan_bias, -1.0, 1.0) * pan_range * 0.50


def apply_directional_lead_space(target_anchor_x, attack_direction, crop_w, img_w, fast_break=False):
    """
    给进攻前方留一点空间。靠近边缘时自动减弱，避免贴边和黑边。
    """
    if attack_direction == 0:
        return target_anchor_x

    lead_ratio = LEAD_SPACE_RATIO_FAST if fast_break else LEAD_SPACE_RATIO_NORMAL
    lead_px = crop_w * lead_ratio * attack_direction

    left_margin = target_anchor_x - crop_w / 2.0
    right_margin = img_w - (target_anchor_x + crop_w / 2.0)

    if attack_direction > 0:
        available = max(0.0, right_margin)
    else:
        available = max(0.0, left_margin)

    gain = clamp(available / max(1.0, crop_w * 0.18), 0.0, 1.0)
    return target_anchor_x + lead_px * gain


def is_pan_settled(current_anchor_x, target_anchor_x, img_w):
    return abs(target_anchor_x - current_anchor_x) <= img_w * PAN_SETTLE_RATIO


def update_director_target_anchor(current_target_x, raw_target_x, img_w, fast_break=False):
    if current_target_x is None:
        return raw_target_x

    delta = raw_target_x - current_target_x
    if abs(delta) <= img_w * PAN_TARGET_DEAD_ZONE:
        return current_target_x

    alpha = FAST_BREAK_TARGET_ALPHA if fast_break else PAN_TARGET_ALPHA
    max_step_ratio = FAST_BREAK_TARGET_MAX_STEP if fast_break else PAN_TARGET_MAX_STEP
    max_step = img_w * max_step_ratio
    eased_step = delta * alpha
    step = clamp(eased_step, -max_step, max_step)

    if abs(step) >= abs(delta):
        return raw_target_x
    return current_target_x + step


def update_anchor_with_inertia(current_anchor_x, anchor_velocity, target_anchor_x, img_w, fast_break=False):
    delta = target_anchor_x - current_anchor_x

    accel_gain = FAST_BREAK_ANCHOR_ACCEL if fast_break else PAN_ANCHOR_ACCEL
    max_velocity_ratio = FAST_BREAK_MAX_VELOCITY if fast_break else PAN_ANCHOR_MAX_VELOCITY
    max_accel_ratio = FAST_BREAK_MAX_ACCEL if fast_break else PAN_ANCHOR_MAX_ACCEL

    desired_velocity = delta * accel_gain
    max_velocity = img_w * max_velocity_ratio
    desired_velocity = clamp(desired_velocity, -max_velocity, max_velocity)

    velocity_delta = desired_velocity - anchor_velocity
    max_accel = img_w * max_accel_ratio
    velocity_delta = clamp(velocity_delta, -max_accel, max_accel)

    anchor_velocity = (anchor_velocity + velocity_delta) * PAN_ANCHOR_DAMPING
    anchor_velocity = clamp(anchor_velocity, -max_velocity, max_velocity)

    if abs(delta) <= img_w * 0.0025 and abs(anchor_velocity) <= img_w * 0.0010:
        return target_anchor_x, 0.0

    next_anchor_x = current_anchor_x + anchor_velocity
    if (target_anchor_x - current_anchor_x) * (target_anchor_x - next_anchor_x) < 0:
        return target_anchor_x, 0.0

    return next_anchor_x, anchor_velocity


def decide_move_with_hysteresis(
    current_anchor_x,
    target_anchor_x,
    img_w,
    prev_move_state,
    move_switch_cooldown,
    move_direction_lock_frames
):
    delta = target_anchor_x - current_anchor_x
    dead_zone = img_w * MOVE_DECISION_DEAD_ZONE_RATIO
    commit_zone = img_w * MOVE_COMMIT_DISTANCE_RATIO

    desired_state = MOVE_STATIONARY
    if prev_move_state == MOVE_STATIONARY:
        if delta > commit_zone:
            desired_state = MOVE_RIGHT
        elif delta < -commit_zone:
            desired_state = MOVE_LEFT
    else:
        if delta > dead_zone:
            desired_state = MOVE_RIGHT
        elif delta < -dead_zone:
            desired_state = MOVE_LEFT

    if move_direction_lock_frames > 0 and prev_move_state in (MOVE_LEFT, MOVE_RIGHT):
        strong_keep = commit_zone * 0.55
        if prev_move_state == MOVE_RIGHT and delta > -strong_keep:
            return prev_move_state, move_switch_cooldown, move_direction_lock_frames - 1
        if prev_move_state == MOVE_LEFT and delta < strong_keep:
            return prev_move_state, move_switch_cooldown, move_direction_lock_frames - 1

    if desired_state == prev_move_state:
        if desired_state in (MOVE_LEFT, MOVE_RIGHT):
            move_direction_lock_frames = max(move_direction_lock_frames, MOVE_DIRECTION_LOCK_FRAMES // 2)
        return desired_state, max(0, move_switch_cooldown - 1), move_direction_lock_frames

    if move_switch_cooldown > 0:
        if abs(delta) < commit_zone * 1.25:
            return prev_move_state, move_switch_cooldown - 1, max(0, move_direction_lock_frames - 1)

    if prev_move_state == MOVE_RIGHT and desired_state == MOVE_LEFT and delta > -commit_zone:
        return MOVE_RIGHT, max(0, move_switch_cooldown - 1), max(0, move_direction_lock_frames - 1)

    if prev_move_state == MOVE_LEFT and desired_state == MOVE_RIGHT and delta < commit_zone:
        return MOVE_LEFT, max(0, move_switch_cooldown - 1), max(0, move_direction_lock_frames - 1)

    new_cooldown = MOVE_SWITCH_COOLDOWN_FRAMES if desired_state != prev_move_state else max(0, move_switch_cooldown - 1)
    new_lock = MOVE_DIRECTION_LOCK_FRAMES if desired_state in (MOVE_LEFT, MOVE_RIGHT) else max(0, move_direction_lock_frames - 1)
    return desired_state, new_cooldown, new_lock


def update_display_move_state(display_state, inner_state, display_pending_state, display_stable_count):
    """
    让左上角 MOVE 显示比内部控制再慢一步，避免文字状态过于敏感。
    """
    if inner_state == display_state:
        return display_state, inner_state, 0

    if inner_state != display_pending_state:
        display_pending_state = inner_state
        display_stable_count = 1
    else:
        display_stable_count += 1

    if display_stable_count >= MOVE_DISPLAY_STABLE_FRAMES:
        return inner_state, inner_state, 0

    return display_state, display_pending_state, display_stable_count


def decide_zoom(main_region, img_w, img_h, current_scale):
    if main_region is None:
        return ZOOM_HOLD, current_scale

    rx1, ry1, rx2, ry2 = main_region
    region_w = rx2 - rx1
    region_h = ry2 - ry1

    width_ratio = region_w / max(1, img_w)
    height_ratio = region_h / max(1, img_h)

    if width_ratio <= 0 or height_ratio <= 0:
        return ZOOM_HOLD, current_scale

    target_scale = min(1.02 / width_ratio, 1.18 / height_ratio)
    target_scale = clamp(target_scale, BASE_CAMERA_SCALE, MAX_CAMERA_SCALE)

    if target_scale > current_scale + 0.02:
        return ZOOM_IN, target_scale
    elif target_scale < current_scale - 0.02:
        return ZOOM_OUT, target_scale
    else:
        return ZOOM_HOLD, target_scale


def pad_region_for_zoom(region, img_w, img_h):
    if region is None:
        return None

    x1, y1, x2, y2 = region
    pad_x = img_w * ZOOM_REGION_PAD_X
    pad_y = img_h * ZOOM_REGION_PAD_Y
    return [
        clamp(x1 - pad_x, 0, img_w),
        clamp(y1 - pad_y, 0, img_h),
        clamp(x2 + pad_x, 0, img_w),
        clamp(y2 + pad_y, 0, img_h),
    ]


def quantize_zoom_target(target_scale):
    steps = round((target_scale - BASE_CAMERA_SCALE) / ZOOM_TARGET_STEP)
    return clamp(BASE_CAMERA_SCALE + steps * ZOOM_TARGET_STEP, BASE_CAMERA_SCALE, MAX_CAMERA_SCALE)


def build_crop(anchor_x, scale_value, img_w, img_h, out_ratio, focus_y=None):
    crop_w, crop_h = get_crop_size(scale_value, img_w, img_h, out_ratio)
    crop_cx = anchor_x
    crop_cy = img_h * VERTICAL_HOME_Y_RATIO if focus_y is None else focus_y
    return [crop_cx, crop_cy, crop_w, crop_h]

def init_overlay_director_state(frame_w):
    current_anchor_x = frame_w * 0.50
    return {
        # current_window：当前已经稳定生效的窗口。
        # target_window：这一帧根据算法判断出来的目标窗口。
        # pending_window：还在观察、尚未正式切换的候选窗口。
        "current_window": WINDOW_CENTER,
        "target_window": WINDOW_CENTER,
        "pending_window": WINDOW_CENTER,
        # window_stable_count：候选窗口已连续稳定多少帧。
        # window_dwell_frames：当前窗口已经停留多少帧。
        # window_switch_cooldown：窗口切换后的冷却帧数。
        "window_stable_count": 0,
        "window_dwell_frames": WINDOW_MIN_DWELL_FRAMES,
        "window_switch_cooldown": 0,
        # scale_value：当前镜头缩放倍率。
        "scale_value": BASE_CAMERA_SCALE,
        # prev_zoom_region：上一帧用于 zoom 判断的区域。
        # zoom_region_stable_count：zoom 区域已经稳定多少帧。
        # pan_settled_count：pan 已经稳定多少帧。
        "prev_zoom_region": None,
        "zoom_region_stable_count": 0,
        "pan_settled_count": 0,
        # zoom_commit_frames / zoom_cooldown_frames：zoom 提交和冷却计数。
        # prev_zoom_target_scale：上一轮已经提交的 zoom 目标倍率。
        "zoom_commit_frames": 0,
        "zoom_cooldown_frames": 0,
        "prev_zoom_target_scale": BASE_CAMERA_SCALE,
        # current_anchor_x：当前镜头水平锚点。
        # current_anchor_v：当前镜头水平速度。
        # director_target_anchor_x：导播层希望镜头去追的目标锚点。
        "current_anchor_x": current_anchor_x,
        "current_anchor_v": 0.0,
        "director_target_anchor_x": current_anchor_x,
        # prev_density_pan_bias：上一帧的 pan 偏置。
        "prev_density_pan_bias": 0.0,
        # pan_side_hold_frames：侧边趋势已经持续多少帧。
        # pan_side_hold_sign：当前侧边趋势方向，1 表示右，-1 表示左。
        "pan_side_hold_frames": 0,
        "pan_side_hold_sign": 0,
        # zoom_freeze_frames：暂时冻结 zoom 的剩余帧数。
        # prev_target_anchor_x：上一帧目标锚点。
        "zoom_freeze_frames": 0,
        "prev_target_anchor_x": current_anchor_x,
        # move_display_from_window / move_display_to_window：MOVE 动画起止窗口。
        # move_display_progress：MOVE 动画进度，范围 0~1。
        # move_display_text：左上角显示的 MOVE 文本。
        "move_display_from_window": WINDOW_CENTER,
        "move_display_to_window": WINDOW_CENTER,
        "move_display_progress": 0.0,
        "move_display_text": MOVE_STATIONARY,
        # current_box_scale：当前导播框整体缩放比例。
        # box_scale_from / box_scale_to：本轮缩放动画的起点和终点比例。
        # box_scale_progress：本轮缩放动画进度。
        # box_scale_mode：当前缩放模式。
        "current_box_scale": 1.0,
        "box_scale_from": 1.0,
        "box_scale_to": 1.0,
        "box_scale_progress": 1.0,
        "box_scale_mode": "center",
        # display_box_rect：显示层平滑后的导播框矩形，格式为 (x1, y1, x2, y2)。
        "display_box_rect": None,
        # arrival_hold_frames：追到新热点后，还要驻留多少帧。
        # arrival_hold_anchor_x：驻留时锁定的锚点。
        # arrival_hold_pan_bias：驻留时锁定的偏置。
        # arrival_hold_direction：驻留时的进攻方向。
        "arrival_hold_frames": 0,
        "arrival_hold_anchor_x": None,
        "arrival_hold_pan_bias": None,
        "arrival_hold_direction": 0,
        # landing_hold_frames：镜头真正到位后，再额外稳住多少帧。
        # landing_hold_anchor_x：落位保持时使用的锚点。
        "landing_hold_frames": 0,
        "landing_hold_anchor_x": None,
    }

def update_overlay_director_state(analysis, frame_w, frame_h, out_ratio, state):
    if analysis["triggered_fast_break"]:
        state["zoom_freeze_frames"] = max(state["zoom_freeze_frames"], FAST_BREAK_ZOOM_FREEZE_FRAMES)

    # raw_density_pan_bias：由热点位置直接换算出的原始 pan 偏置。
    # density_trend：偏置变化趋势项，用于增强连续向左/向右的倾向。
    # pan_decision_bias：最终用于导播决策的偏置。
    raw_density_pan_bias = focus_to_pan_bias(analysis["predicted_hotspot_x"], frame_w)
    raw_density_pan_bias = scale_pan_bias_by_group_size(raw_density_pan_bias, len(analysis["pan_group"]))

    density_trend = clamp(
        (raw_density_pan_bias - state["prev_density_pan_bias"]) * PAN_TREND_GAIN,
        -PAN_TREND_LIMIT,
        PAN_TREND_LIMIT,
    )

    trend_gain = 0.65 if analysis["fast_break_active"] else 0.20
    pan_decision_bias = clamp(raw_density_pan_bias + density_trend * trend_gain, -1.0, 1.0)

    if abs(raw_density_pan_bias) < PAN_CENTER_DEAD_BIAS:
        if analysis["hold_after_break_active"] or state["arrival_hold_frames"] > 0 or state["landing_hold_frames"] > 0:
            pan_decision_bias = state["prev_density_pan_bias"] * 0.96
        else:
            pan_decision_bias = 0.0

    pan_sign = (
        1 if pan_decision_bias > PAN_CENTER_DEAD_BIAS
        else -1 if pan_decision_bias < -PAN_CENTER_DEAD_BIAS
        else 0
    )

    if pan_sign == 0:
        state["pan_side_hold_frames"] = 0
        state["pan_side_hold_sign"] = 0
    elif pan_sign == state["pan_side_hold_sign"]:
        state["pan_side_hold_frames"] += 1
    else:
        state["pan_side_hold_sign"] = pan_sign
        state["pan_side_hold_frames"] = 1

    if state["pan_side_hold_frames"] >= PAN_SIDE_HOLD_DECAY_FRAMES and abs(pan_decision_bias) < WINDOW_ENTER_BIAS:
        pan_decision_bias *= PAN_SIDE_HOLD_DECAY_SCALE

    pan_decision_bias = clamp(pan_decision_bias, -PAN_SIDE_BIAS_LIMIT, PAN_SIDE_BIAS_LIMIT)

    state["target_window"] = choose_window_by_group_density(
        analysis["pan_group"],
        analysis["window_focus_x"],
        analysis["window_focus_y"],
        frame_w,
        frame_h,
        state["current_window"],
    )
    previous_window = state["current_window"]
    (
        state["current_window"],
        state["pending_window"],
        state["window_stable_count"],
        state["window_dwell_frames"],
        switched,
        state["window_switch_cooldown"],
    ) = update_window_with_hysteresis(
        state["current_window"],
        state["target_window"],
        state["pending_window"],
        state["window_stable_count"],
        state["window_dwell_frames"],
        state["window_switch_cooldown"],
    )

    if switched:
        state["zoom_freeze_frames"] = max(state["zoom_freeze_frames"], PAN_FREEZE_FRAMES_AFTER_SWITCH)

    if state["arrival_hold_frames"] > 0 and state["arrival_hold_pan_bias"] is not None:
        target_pan_bias = state["arrival_hold_pan_bias"]
    else:
        if abs(pan_decision_bias) <= RECENTER_FAST_BIAS:
            if analysis["hold_after_break_active"] or state["landing_hold_frames"] > 0:
                target_pan_bias = state["prev_density_pan_bias"]
            else:
                target_pan_bias = smooth_value(state["prev_density_pan_bias"], 0.0, alpha=RECENTER_FAST_ALPHA)
        elif abs(pan_decision_bias) < PAN_CENTER_DEAD_BIAS:
            if analysis["hold_after_break_active"] or state["landing_hold_frames"] > 0:
                target_pan_bias = state["prev_density_pan_bias"]
            else:
                target_pan_bias = smooth_value(state["prev_density_pan_bias"], 0.0, alpha=PAN_CENTER_RETURN_ALPHA)
        else:
            pan_target_alpha = (
                FAST_BREAK_TARGET_ALPHA
                if analysis["fast_break_active"]
                else PAN_DENSITY_TARGET_ALPHA * clamp(analysis["density_confidence"], 0.55, 1.0)
            )
            target_pan_bias = smooth_value(
                state["prev_density_pan_bias"],
                pan_decision_bias,
                alpha=pan_target_alpha,
            )

    target_pan_bias = clamp(target_pan_bias, -1.0, 1.0)
    state["prev_density_pan_bias"] = target_pan_bias

    # crop_w_for_lead：当前 scale 下的理论裁切宽度，用于计算横向留白。
    # guidance_from_window / guidance_to_window / guidance_progress：当前窗口引导动画的起点、终点和进度。
    crop_w_for_lead, _ = get_crop_size(state["scale_value"], frame_w, frame_h, out_ratio)
    guidance_from_window = state["move_display_from_window"]
    guidance_to_window = state["move_display_to_window"]
    guidance_progress = state["move_display_progress"] if guidance_from_window != guidance_to_window else 0.0

    # raw_target_anchor_x：未做导播层惯性前的目标锚点。
    # target_anchor_x：经过导播惯性和驻留逻辑后的最终目标锚点。
    raw_target_anchor_x = get_pan_bias_center(target_pan_bias, frame_w, crop_w_for_lead)
    raw_target_anchor_x = apply_directional_lead_space(
        raw_target_anchor_x,
        analysis["attack_direction"],
        crop_w_for_lead,
        frame_w,
        fast_break=analysis["fast_break_active"],
    )
    raw_target_anchor_x = apply_window_guidance(
        raw_target_anchor_x,
        guidance_from_window,
        guidance_to_window,
        guidance_progress,
        frame_w,
        crop_w_for_lead,
        fast_break=analysis["fast_break_active"],
    )

    target_anchor_x = update_director_target_anchor(
        state["director_target_anchor_x"],
        raw_target_anchor_x,
        frame_w,
        fast_break=analysis["fast_break_active"],
    )

    if state["arrival_hold_frames"] <= 0:
        arrival_distance = abs(target_anchor_x - state["current_anchor_x"])
        if (
            arrival_distance <= frame_w * ARRIVAL_HOLD_DISTANCE_RATIO
            and analysis["attack_direction"] != 0
            and abs(target_pan_bias) >= ARRIVAL_HOLD_DIRECTION_TOLERANCE
        ):
            state["arrival_hold_frames"] = ARRIVAL_HOLD_FRAMES
            state["arrival_hold_anchor_x"] = target_anchor_x
            state["arrival_hold_pan_bias"] = target_pan_bias
            state["arrival_hold_direction"] = analysis["attack_direction"]

    if state["arrival_hold_frames"] > 0:
        state["arrival_hold_frames"] -= 1

        reverse_signal = (
            (state["arrival_hold_direction"] > 0 and target_pan_bias < -ARRIVAL_HOLD_DIRECTION_TOLERANCE)
            or (state["arrival_hold_direction"] < 0 and target_pan_bias > ARRIVAL_HOLD_DIRECTION_TOLERANCE)
        )

        if reverse_signal:
            state["arrival_hold_frames"] = 0
            state["arrival_hold_anchor_x"] = None
            state["arrival_hold_pan_bias"] = None
            state["arrival_hold_direction"] = 0
        elif state["arrival_hold_anchor_x"] is not None:
            target_anchor_x = smooth_value(
                target_anchor_x,
                state["arrival_hold_anchor_x"],
                alpha=ARRIVAL_HOLD_BLEND,
            )

    if state["landing_hold_frames"] <= 0:
        landing_distance = abs(state["current_anchor_x"] - target_anchor_x)
        if landing_distance <= frame_w * LANDING_HOLD_DISTANCE_RATIO:
            state["landing_hold_frames"] = LANDING_HOLD_FRAMES
            state["landing_hold_anchor_x"] = target_anchor_x

    if state["landing_hold_frames"] > 0:
        state["landing_hold_frames"] -= 1
        if state["landing_hold_anchor_x"] is not None:
            target_anchor_x = smooth_value(
                target_anchor_x,
                state["landing_hold_anchor_x"],
                alpha=LANDING_HOLD_BLEND,
            )

    state["director_target_anchor_x"] = target_anchor_x

    # old_dir：上一帧目标方向。
    # new_dir：当前目标方向。
    old_dir = target_anchor_x - state["prev_target_anchor_x"]
    new_dir = target_anchor_x - state["current_anchor_x"]
    if old_dir * new_dir < 0:
        state["current_anchor_v"] *= PAN_REVERSAL_DAMPING
    state["prev_target_anchor_x"] = target_anchor_x

    move_target_window = resolve_move_target_window(
        state["current_window"],
        state["pending_window"],
        state["target_window"],
    )
    if move_target_window == state["current_window"] and abs(new_dir) <= frame_w * 0.010:
        state["current_anchor_v"] *= 0.84

    state["current_anchor_x"], state["current_anchor_v"] = update_anchor_with_inertia(
        state["current_anchor_x"],
        state["current_anchor_v"],
        target_anchor_x,
        frame_w,
        fast_break=analysis["fast_break_active"],
    )

    (
        state["move_display_from_window"],
        state["move_display_to_window"],
        state["move_display_progress"],
        state["move_display_text"],
    ) = update_move_progress_display(
        state["move_display_from_window"],
        state["move_display_to_window"],
        state["move_display_progress"],
        previous_window,
        state["current_window"],
        switched,
    )

    if (
        state["move_display_from_window"] != state["move_display_to_window"]
        and state["move_display_to_window"] == WINDOW_CENTER
    ):
        if state["box_scale_mode"] != "move_to_center":
            state["box_scale_from"] = state["current_box_scale"]
            state["box_scale_to"] = 1.0
            state["box_scale_mode"] = "move_to_center"
        state["box_scale_progress"] = state["move_display_progress"]
        state["current_box_scale"] = interpolate_box_scale(
            state["box_scale_from"],
            state["box_scale_to"],
            state["box_scale_progress"],
        )
    elif (
        state["move_display_from_window"] == state["move_display_to_window"]
        and state["current_window"] in (WINDOW_LEFT, WINDOW_RIGHT)
    ):
        stationary_mode = f"side_stationary_{state['current_window']}"
        if state["box_scale_mode"] != stationary_mode:
            state["box_scale_from"] = state["current_box_scale"]
            state["box_scale_to"] = get_side_box_scale()
            state["box_scale_progress"] = 0.0
            state["box_scale_mode"] = stationary_mode
        state["box_scale_progress"] = min(1.0, state["box_scale_progress"] + MOVE_PROGRESS_MAX_STEP)
        state["current_box_scale"] = interpolate_box_scale(
            state["box_scale_from"],
            state["box_scale_to"],
            state["box_scale_progress"],
        )
    else:
        state["current_box_scale"] = 1.0
        state["box_scale_from"] = 1.0
        state["box_scale_to"] = 1.0
        state["box_scale_progress"] = 1.0
        state["box_scale_mode"] = "center"

    target_box_rect = get_window_guide_box_rect(
        frame_w,
        frame_h,
        state["move_display_from_window"],
        state["move_display_to_window"],
        state["move_display_progress"],
        box_scale=state["current_box_scale"],
    )
    state["display_box_rect"] = smooth_box_rect(state["display_box_rect"], target_box_rect)

    # pan_settled：当前镜头是否已经基本追上目标锚点。
    pan_settled = is_pan_settled(state["current_anchor_x"], target_anchor_x, frame_w)
    state["pan_settled_count"] = state["pan_settled_count"] + 1 if pan_settled else 0

    zoom_region_input = pad_region_for_zoom(analysis["main_region"], frame_w, frame_h)
    zoom_region = smooth_region(state["prev_zoom_region"], zoom_region_input, alpha=0.05)

    if region_change_ratio(state["prev_zoom_region"], zoom_region, frame_w, frame_h) <= 0.010:
        state["zoom_region_stable_count"] += 1
    else:
        state["zoom_region_stable_count"] = 0

    state["prev_zoom_region"] = zoom_region

    if state["zoom_cooldown_frames"] > 0:
        state["zoom_cooldown_frames"] -= 1

    zoom_ready = (
        not analysis["fast_break_active"]
        and not analysis["hold_after_break_active"]
        and state["arrival_hold_frames"] <= 0
        and state["landing_hold_frames"] <= 0
        and len(analysis["main_group"]) >= 4
        and analysis["density_confidence"] >= 0.60
        and state["pan_settled_count"] >= ZOOM_AFTER_PAN_SETTLE_FRAMES
        and state["zoom_region_stable_count"] >= ZOOM_REGION_STABLE_FRAMES
    )

    if state["zoom_freeze_frames"] > 0:
        target_scale = state["scale_value"]
        state["prev_zoom_target_scale"] = state["scale_value"]
        state["zoom_freeze_frames"] -= 1
        state["zoom_commit_frames"] = 0
        state["zoom_cooldown_frames"] = max(state["zoom_cooldown_frames"], ZOOM_AFTER_PAN_SETTLE_FRAMES)
    elif zoom_ready:
        _, raw_target_scale = decide_zoom(zoom_region, frame_w, frame_h, state["scale_value"])
        raw_target_scale = quantize_zoom_target(raw_target_scale)

        if (
            state["zoom_cooldown_frames"] <= 0
            and state["zoom_commit_frames"] <= 0
            and abs(raw_target_scale - state["scale_value"]) >= ZOOM_TARGET_DEAD_BAND
        ):
            state["prev_zoom_target_scale"] = raw_target_scale
            state["zoom_commit_frames"] = ZOOM_COMMIT_FRAMES
            state["zoom_cooldown_frames"] = ZOOM_COOLDOWN_FRAMES

        if state["zoom_commit_frames"] > 0:
            target_scale = state["prev_zoom_target_scale"]
            state["zoom_commit_frames"] -= 1
        else:
            target_scale = state["scale_value"]
            state["prev_zoom_target_scale"] = state["scale_value"]
    else:
        target_scale = state["scale_value"]

    state["scale_value"] = smooth_value(state["scale_value"], target_scale, alpha=ZOOM_ALPHA)

    crop_w_after_zoom, _ = get_crop_size(state["scale_value"], frame_w, frame_h, out_ratio)
    state["current_anchor_x"] = clamp(
        state["current_anchor_x"],
        crop_w_after_zoom / 2.0,
        frame_w - crop_w_after_zoom / 2.0,
    )
    state["director_target_anchor_x"] = clamp(
        state["director_target_anchor_x"],
        crop_w_after_zoom / 2.0,
        frame_w - crop_w_after_zoom / 2.0,
    )
    state["prev_target_anchor_x"] = clamp(
        state["prev_target_anchor_x"],
        crop_w_after_zoom / 2.0,
        frame_w - crop_w_after_zoom / 2.0,
    )

    return {
        "current_window": state["current_window"],
        "move_display_from_window": state["move_display_from_window"],
        "move_display_to_window": state["move_display_to_window"],
        "move_display_progress": state["move_display_progress"],
        "move_display_text": state["move_display_text"],
        "current_box_scale": state["current_box_scale"],
        "display_box_rect": state["display_box_rect"],
    }


def render_overlay_outputs(
    frame,
    boxes_info,
    analysis,
    director,
    draw_debug_text,
    draw_debug_boxes,
    program_out_w,
    program_out_h,
):
    program_frame = extract_program_frame(
        frame,
        director["move_display_from_window"],
        director["move_display_to_window"],
        director["move_display_progress"],
        program_out_w,
        program_out_h,
        box_scale=director["current_box_scale"],
        box_rect=director["display_box_rect"],
    )
    # out_frame：叠加调试信息后的预览画面。
    out_frame = frame.copy()

    if draw_debug_boxes:
        out_frame = draw_boxes_on_frame(out_frame, boxes_info)
        out_frame = draw_window_guide_box(
            out_frame,
            director["move_display_from_window"],
            director["move_display_to_window"],
            director["move_display_progress"],
            box_scale=director["current_box_scale"],
            box_rect=director["display_box_rect"],
        )

    if draw_debug_text:
        cv2.putText(out_frame, f"Window: {director['current_window']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(out_frame, f"Move: {director['move_display_text']}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(out_frame, f"Track Y: {int(frame.shape[0] * WINDOW_TRACK_LINE_Y_RATIO)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(out_frame, f"Density: {analysis['density_confidence']:.2f}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(out_frame, f"Battle: {len(analysis['main_group'])} / Persons: {analysis['detections_count']}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(out_frame, f"Hotspot X: {int(analysis['window_focus_x'])}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return out_frame, program_frame
