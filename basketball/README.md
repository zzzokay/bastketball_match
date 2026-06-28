# Basketball 双目拼接、球员检测与自动运镜项目

本项目运行在 RK3588 / ELF2 / Linux 环境，用两路 USB 摄像头覆盖篮球场左右视角，经过单目标定、双目极线校正、宽幅拼接、RKNN 球员检测和自动运镜，输出完整宽幅画面或 1920x1080 导播视口。

核心目标：

1. 采集左右两路 1920x1080@30FPS USB 摄像头画面。
2. 分别完成左右摄像头单目标定，得到内参和畸变参数。
3. 离线生成 raw 图到 rectified 图的双目 remap 查找表。
4. 离线估计左右画面的重叠宽度、垂直偏移和 alpha 融合参数。
5. 实时完成双目矫正、拼接、调试显示和保存。
6. 使用 RKNN 模型检测球员，并把检测框从 raw 坐标映射到宽幅坐标。
7. 根据球员分布生成 1920x1080 自动运镜视口。
8. 支持保存照片、录制宽幅视频、录制 1920x1080 视口视频和调试输出。

---

## 1. 整体流程

```text
左右 USB 摄像头
    │
    ├─ 阶段 A：单目标定
    │     calib_usb2_camera.py      左相机 /dev/video41 -> camera_usb2_calib.npz
    │     calib_usb_camera.py       右相机 /dev/video43 -> camera_calib.npz
    │
    ├─ 阶段 B：双目标定与 remap 生成
    │     offline_build_stereo_rectify_maps.py
    │       capture                 采集左右双目标定图
    │       build                   生成 raw -> rectified remap
    │       test-image              测试 remap 极线效果
    │       capture-rectified       采集最终 rectified 左右图
    │
    ├─ 阶段 C：拼接参数估计
    │     offline_estimate_stitch_params.py
    │       估计 overlap_px / vertical_offset / blend_width / alpha_mask
    │
    ├─ 阶段 D：实时拼接与调试
    │     realtime_stereo_stitch.py
    │     realtime_stereo_stitch_1920_view.py
    │
    ├─ 阶段 E：照片 / 视频留存
    │     save_wide_stitch_photo_to_tf.py
    │     save_wide_and_view_photo_to_tf.py
    │     record_wide_and_view_video_to_tf.py
    │     record_select_h264_rkmpp_video.py
    │     record_select_h264_rkmpp_rectified_option.py
    │
    └─ 阶段 F：RKNN 检测与自动运镜
          infer_person.py
          camera_movement/dual_rknn_wide_coords_display_alternating.py
          camera_movement/dual_rknn_director_view1920_fast.py
          camera_movement/dual_rknn_director_view1920_record_local.py
```

推荐主线：

```text
calib_usb2_camera.py + calib_usb_camera.py
        ↓
offline_build_stereo_rectify_maps.py
        ↓
offline_estimate_stitch_params.py
        ↓
realtime_stereo_stitch_1920_view.py / realtime_stereo_stitch.py
        ↓
camera_movement/dual_rknn_director_view1920_fast.py
```

---

## 2. 硬件与默认参数

| 项目 | 默认值 | 说明 |
|---|---:|---|
| 左摄像头 | `/dev/video41` | USB2 摄像头，通常对应 `calib_usb2_camera.py` |
| 右摄像头 | `/dev/video43` | USB 摄像头，通常对应 `calib_usb_camera.py` |
| 分辨率 | `1920x1080` | 标定、remap、实时运行建议保持一致 |
| 帧率 | `30` | 双路 USB 采集目标帧率 |
| 采集格式 | `MJPG` | 降低 USB 带宽压力 |
| 标定板 | `11 x 8` 内角点 | 对应 12 x 9 方格 |
| 方格边长 | `22 mm` | 当前脚本默认值 |
| 平台 | RK3588 / ELF2 / Linux | 使用 V4L2、OpenCV、RKNNLite |

查看设备：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video41 --list-formats-ext
v4l2-ctl -d /dev/video43 --list-formats-ext
v4l2-ctl -d /dev/video41 --list-ctrls-menus
v4l2-ctl -d /dev/video43 --list-ctrls-menus
```

---

## 3. 依赖

### Python 依赖

```bash
python3 -m pip install numpy opencv-python
```

RKNN 推理相关脚本还需要当前板端可用的 RKNNLite：

```python
from rknnlite.api import RKNNLite
```

如果使用视频录制脚本，需要系统中有 `ffmpeg`，并且 RK3588 镜像支持 `h264_rkmpp` 硬件编码。

### 系统工具

```bash
sudo apt install v4l-utils ffmpeg
```

检查 OpenCV：

```bash
python3 - <<'PY'
import cv2, numpy as np
print('cv2:', cv2.__version__)
print('numpy:', np.__version__)
PY
```

---

## 4. 目录与关键文件

```text
basketball/
├── README.md
├── calib_usb_camera.py                         # 右相机单目标定与实时畸变矫正
├── calib_usb2_camera.py                        # 左相机单目标定与实时畸变矫正
├── camera_calib.npz                            # 右相机单目标定结果
├── camera_usb2_calib.npz                       # 左相机单目标定结果
│
├── calib_images/                               # 右相机标定图和角点预览图
├── calib_usb2_images/                          # 左相机标定图和角点预览图
│
├── offline_build_stereo_rectify_maps.py         # 双目标定、remap 生成、remap 测试
├── offline_build_stereo_rectify_maps/           # remap、拼接参数、双目标定图、调试图
│   ├── stereo_rectify_maps_wide.npz
│   ├── stereo_rectify_maps_wide_good.npz
│   ├── stitch_params.npz
│   └── stitch_params_good.npz
│
├── offline_estimate_stitch_params.py            # 估计 overlap / vertical_offset / alpha mask
├── realtime_stereo_stitch.py                    # 实时输出完整宽幅拼接图
├── realtime_stereo_stitch_1920_view.py          # 实时输出 1920x1080 裁切视口
├── realtime_stitch_debug/                       # 实时调试保存图
│
├── infer_person.py                              # 单路 RKNN 人体检测最小示例
├── model/                                       # RKNN / ONNX 模型和 labels.txt
│   ├── labels.txt
│   ├── basketball_player.onnx
│   ├── basketball_player.rknn
│   ├── basketball_player_2.1.0.rknn
│   ├── basketball_player_fp_2.1.0.rknn
│   └── best_2.rknn
│
├── camera_movement/                             # 双路检测、坐标映射、自动运镜
│   ├── dual_rknn_wide_coords_display_alternating.py
│   ├── dual_rknn_director_view1920_fast.py
│   ├── dual_rknn_director_view1920_record_local.py
│   ├── predict1_weighted.py
│   └── predict1_director.py
│
├── save_wide_stitch_photo_to_tf.py              # 保存完整宽幅 JPG
├── save_wide_and_view_photo_to_tf.py            # 保存宽幅 JPG + 1920 视口 JPG
├── record_wide_and_view_video_to_tf.py          # 录制 MP4 宽幅 + 1920 视口
├── record_wide_and_view_video_to_tf_avi.py      # 录制 AVI/MJPG 版本
├── record_select_h264_rkmpp_video.py            # 多路选择录制，H.264/RKMPP
├── record_select_h264_rkmpp_rectified_option.py # 可选 raw/rectified 单目源的录制版本
└── auto_start_left_right.patch                  # 自动启动相关补丁记录
```

说明：

- `*_good.npz` 通常表示人工确认效果较稳定的参数文件，实时阶段优先使用。
- `*.bak*`、`camera_movement/_archive_unused_*` 属于备份或旧实验版本，不建议作为主流程入口。
- 调试目录如 `realtime_stitch_debug/`、`stereo_rectify_debug/` 可删除后重新生成。

---

## 5. 脚本在流程中的作用

### 5.1 标定脚本

| 脚本 | 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `calib_usb2_camera.py` | 单目标定 | 左相机 `/dev/video41`、棋盘格 | `calib_usb2_images/`、`camera_usb2_calib.npz` | 采集左相机棋盘格图，计算左相机内参、畸变系数，也可实时查看畸变矫正效果。 |
| `calib_usb_camera.py` | 单目标定 | 右相机 `/dev/video43`、棋盘格 | `calib_images/`、`camera_calib.npz` | 采集右相机棋盘格图，计算右相机内参、畸变系数，也可实时查看畸变矫正效果。 |

两个脚本都有三种模式：

```text
--mode capture      采集标定图片
--mode calibrate    根据 calib_*.jpg 计算内参和畸变参数
--mode undistort    实时预览畸变矫正效果
```

常用按键：

| 按键 | 作用 |
|---|---|
| `c` | 保存检测到棋盘格角点的标定图 |
| `s` | 直接保存当前原图 |
| `o` | undistort 模式下切换原图 / 矫正图对比 |
| `q` / `Esc` | 退出 |

### 5.2 双目 remap 生成脚本

| 脚本 | 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `offline_build_stereo_rectify_maps.py` | 双目标定 / remap | `camera_usb2_calib.npz`、`camera_calib.npz`、左右双目标定图 | `stereo_rectify_maps*.npz`、角点调试图、极线调试图 | 采集双目标定图，运行 stereoCalibrate + stereoRectify，生成实时阶段使用的 raw -> rectified remap 查找表。 |

主要模式：

| 模式 | 作用 | 典型输出 |
|---|---|---|
| `capture` | 同时读取左右摄像头，按 `s` 保存已单目矫正的双目标定图 | `offline_build_stereo_rectify_maps/images/Left/`、`Right/` |
| `build` | 检测双目标定图角点，计算双目外参和 remap 表 | `stereo_rectify_maps_wide.npz` |
| `test-image` | 用一对图片测试 remap 效果，保存水平极线调试图 | `stereo_rectify_debug/test_*` |
| `capture-rectified` | 使用已生成 remap 实时采集最终 rectified 左右图 | `finish_stereo_rectify/Left/`、`Right/` |

关键点：

- 该脚本保存最终 `raw -> rectified` map，实时阶段只需要一次 `cv2.remap`。
- `capture` 模式默认保存“已单目畸变矫正图”，用于后续 `build` 检测棋盘格。
- 如果右图极线方向异常，可尝试 `--right-corner-transform orig/flip_ud/flip_lr/rot180`。
- 大夹角双摄通常使用 `--rectify-alpha 1.0` 保留视野，黑边后续再裁剪。

### 5.3 拼接参数估计脚本

| 脚本 | 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `offline_estimate_stitch_params.py` | 离线拼接参数估计 | remap 文件、一对左右图片 | `stitch_params*.npz`、`stitched_preview.jpg`、`debug_rois.jpg`、`alpha_mask.png` | 估计左右图重叠宽度、上下偏移、融合宽度和最终 ROI，把这些参数保存给实时拼接脚本。 |

支持三种输入类型：

| 输入图片类型 | 参数 |
|---|---|
| 摄像头 raw 原图 | 不加额外参数 |
| 已单目矫正图 | `--input-already-undistorted` |
| 已双目极线矫正图 | `--input-already-rectified` |

重要：如果输入已经是 `left_rect/right_rect` 这类 rectified 图，必须加 `--input-already-rectified`，否则会重复 remap，导致 `overlap_px` 和 `vertical_offset` 错误。

可自动估计，也可手动覆盖：

```text
--manual-overlap            手动指定左右重叠宽度
--manual-vertical-offset    手动指定右图相对左图的上下偏移
--blend-width               真正 alpha 渐变融合区域宽度
```

调参经验：

| 现象 | 优先调整 |
|---|---|
| 左右画面接不上 | `overlap_px` |
| 上下错位 | `vertical_offset` |
| 接缝太硬 | 增大 `blend_width` |
| 接缝重影明显 | 减小 `blend_width`，或调整 runtime seam/shift |

### 5.4 实时拼接脚本

| 脚本 | 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `realtime_stereo_stitch.py` | 实时拼接 | 左右摄像头、`stereo_rectify_maps*.npz`、`stitch_params*.npz` | 完整宽幅拼接显示、调试图 | 按离线参数完成双路 remap、裁剪、平移、颜色补偿和 alpha 融合。 |
| `realtime_stereo_stitch_1920_view.py` | 实时 1920 视口 | 同上 | 1920x1080 裁切视口 | 在完整拼接逻辑基础上输出固定大小视口，支持 runtime seam、blend、右图偏移微调和线程采集。 |

常用按键：

| 按键 | 作用 |
|---|---|
| `q` / `Esc` | 退出 |
| `s` | 保存当前帧调试图 |
| `b` | 开启 / 关闭颜色补偿 |
| `l` | 开启 / 关闭水平参考线 |
| `r` | 开启 / 关闭左右 rectified 调试窗口 |

调试图一般保存在：

```text
realtime_stitch_debug/
```

包含 raw 图、rectified 图、完整拼接图和 1920 视口图。

### 5.5 RKNN 检测与自动运镜脚本

| 脚本 | 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `infer_person.py` | 单路检测验证 | 单个 USB 摄像头、RKNN 模型 | 单路检测显示 | 最小 RKNN 人体检测示例，用于确认模型、标签、摄像头和 RKNNLite 是否正常。 |
| `camera_movement/dual_rknn_wide_coords_display_alternating.py` | 双路检测 + 宽图坐标 | 左右摄像头、RKNN 模型、remap、stitch 参数 | 完整宽幅图检测框显示 | 左右交替调用单 RKNN，避免双路同时抢 NPU；把 raw 检测框映射到 rectified 再映射到 wide 坐标。 |
| `camera_movement/dual_rknn_director_view1920_fast.py` | 自动运镜实时输出 | 双路检测结果、remap、stitch 参数、运镜算法 | 1920x1080 导播视口 | 检测低频、运镜高频；不生成完整宽幅图，直接根据 crop 位置 remap 出 1920x1080 view，提高最终输出 FPS。 |
| `camera_movement/dual_rknn_director_view1920_record_local.py` | 自动运镜 + 本地录制 | 同上 | 1920x1080 导播视口和本地录像 | 在 fast 版本基础上增加本地录制能力。 |
| `camera_movement/predict1_weighted.py` | 运镜算法组件 | 宽图人物框 / 球框 | 关注区域、权重、热点 | 分析人物框、球框、主战区、密度聚类和关注点。 |
| `camera_movement/predict1_director.py` | 运镜算法组件 | `predict1_weighted.py` 的分析结果 | 导播窗口状态 | 负责窗口切换、滞回、平滑、引导框和最终视口决策。 |

检测链路：

```text
左右 raw 图
    ↓
RKNN 在 raw 图上检测 person
    ↓
raw 检测框通过 remap 反查表映射到 rectified 坐标
    ↓
rectified 坐标按 stitch 参数映射到完整 wide 坐标
    ↓
predict1_weighted.py 分析人物分布
    ↓
predict1_director.py 选择导播窗口
    ↓
直接 remap 输出 1920x1080 view
```

### 5.6 拍照与录像脚本

| 脚本 | 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `save_wide_stitch_photo_to_tf.py` | 拍照 | 左右摄像头、remap、stitch 参数 | 完整宽幅 JPG | 保存当前双目拼接后的完整宽图，用于检查拼接、接缝、偏移和参数效果。 |
| `save_wide_and_view_photo_to_tf.py` | 拍照 | 同上 | 宽幅 JPG + 1920x1080 视口 JPG | 同时保存完整宽图和裁切视口，便于对比全景和最终输出。 |
| `record_wide_and_view_video_to_tf.py` | 录像 | 同上 | 宽幅 MP4 + 1920x1080 MP4 | 同时录制完整宽幅视频和最终视口视频。 |
| `record_wide_and_view_video_to_tf_avi.py` | 录像 | 同上 | AVI/MJPG 视频 | MJPG/AVI 版本，适合某些编码环境或调试场景。 |
| `record_select_h264_rkmpp_video.py` | 录像 | 同上、FFmpeg h264_rkmpp | left_raw、right_raw、wide、view1920 多路视频 | 按键选择录制单目 raw、宽幅和 1920 视口，独立队列写入，降低主循环阻塞。 |
| `record_select_h264_rkmpp_rectified_option.py` | 录像 | 同上 | left_mono、right_mono、wide、view1920 多路视频 | 单目视频可选 raw 或 rectified，默认更适合保存 rectified 单目结果。 |

多路录制常用按键：

| 按键 | 作用 |
|---|---|
| `1` | 开始 / 停止左单目视频 |
| `2` | 开始 / 停止右单目视频 |
| `3` | 开始 / 停止完整宽幅视频 |
| `4` | 开始 / 停止 1920 视口视频 |
| `a` | 全部开始 / 全部停止 |
| `s` | 保存当前帧 JPG 调试图 |
| `h` | 显示帮助 |
| `q` / `Esc` | 退出 |

---

## 6. 推荐运行步骤

### 6.1 左相机单目标定

```bash
cd /home/elf/work/basketball

python3 calib_usb2_camera.py \
  --mode capture \
  --device /dev/video41 \
  --width 1920 --height 1080 --fps 30

python3 calib_usb2_camera.py \
  --mode calibrate \
  --input-dir /home/elf/work/basketball/calib_usb2_images \
  --output /home/elf/work/basketball/camera_usb2_calib.npz
```

### 6.2 右相机单目标定

```bash
python3 calib_usb_camera.py \
  --mode capture \
  --device /dev/video43 \
  --width 1920 --height 1080 --fps 30

python3 calib_usb_camera.py \
  --mode calibrate \
  --input-dir /home/elf/work/basketball/calib_images \
  --output /home/elf/work/basketball/camera_calib.npz
```

建议每个摄像头采集 30 张左右，覆盖画面中心、四角、不同距离和不同倾角。平均重投影误差越小越好，通常 `< 0.8 px` 可用，`> 1.5 px` 建议重新采集。

### 6.3 采集双目标定图

```bash
python3 offline_build_stereo_rectify_maps.py \
  --mode capture \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
  --right-calib-file /home/elf/work/basketball/camera_calib.npz \
  --capture-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/images \
  --board-cols 11 --board-rows 8 --square-size 22 \
  --width 1920 --height 1080 --fps 30 \
  --display-scale 0.25
```

按 `s` 保存一组左右图，按 `q` 退出。

### 6.4 生成双目 remap

```bash
python3 offline_build_stereo_rectify_maps.py \
  --mode build \
  --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
  --right-calib-file /home/elf/work/basketball/camera_calib.npz \
  --capture-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/images \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
  --board-cols 11 --board-rows 8 --square-size 22 \
  --mono-dist-scale 0.8 \
  --rectify-alpha 1.0 \
  --out-scale 1.15 \
  --right-corner-transform orig \
  --save-build-preview \
  --headless
```

效果确认可用后保存为稳定版本：

```bash
cp offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
   offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz
```

### 6.5 测试 remap 极线效果

```bash
python3 offline_build_stereo_rectify_maps.py \
  --mode test-image \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
  --left-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/images/Left/left_000000.png \
  --right-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/images/Right/right_000000.png \
  --output-dir /home/elf/work/basketball/stereo_rectify_debug \
  --input-already-undistorted \
  --headless
```

检查：

```text
stereo_rectify_debug/test_wide_lines.jpg
```

左右图中同一棋盘格角点应基本落在同一水平线上。

### 6.6 估计拼接参数

如果输入是已经 rectified 的左右图：

```bash
python3 offline_estimate_stitch_params.py \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
  --left-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/finish_stereo_rectify/Left/left_000000.png \
  --right-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/finish_stereo_rectify/Right/right_000000.png \
  --output-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params.npz \
  --output-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_param_debug \
  --input-already-rectified \
  --min-overlap 700 \
  --max-overlap 1200 \
  --max-vertical-offset 100 \
  --search-scale 0.35
```

如果已经知道可用参数，也可以手动指定：

```bash
python3 offline_estimate_stitch_params.py \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
  --left-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/finish_stereo_rectify/Left/left_000000.png \
  --right-image /home/elf/work/basketball/offline_build_stereo_rectify_maps/finish_stereo_rectify/Right/right_000000.png \
  --output-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
  --output-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_param_debug \
  --input-already-rectified \
  --manual-overlap 1010 \
  --manual-vertical-offset 41 \
  --blend-width 60
```

效果确认可用后：

```bash
cp offline_build_stereo_rectify_maps/stitch_params.npz \
   offline_build_stereo_rectify_maps/stitch_params_good.npz
```

### 6.7 实时完整宽幅拼接

```bash
python3 realtime_stereo_stitch.py \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
  --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
  --width 1920 --height 1080 --fps 30 \
  --display-scale 0.15 \
  --disable-color-balance
```

### 6.8 实时 1920x1080 视口

```bash
python3 realtime_stereo_stitch_1920_view.py \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
  --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
  --width 1920 --height 1080 --fps 30 \
  --view-width 1920 \
  --view-height 1080 \
  --crop-x -1 \
  --crop-y -1 \
  --display-scale 0.25 \
  --disable-color-balance \
  --threaded-capture \
  --runtime-seam-x 150 \
  --runtime-blend-width 40 \
  --runtime-right-x-shift 30 \
  --runtime-right-y-shift -5
```

### 6.9 单路 RKNN 检测验证

```bash
python3 infer_person.py \
  --device /dev/video43 \
  --model /home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn \
  --labels /home/elf/work/basketball/model/labels.txt \
  --conf 0.25 \
  --nms 0.45
```

### 6.10 双路 RKNN 检测 + 宽图坐标显示

```bash
cd /home/elf/work/basketball/camera_movement

python3 dual_rknn_wide_coords_display_alternating.py \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --model /home/elf/work/basketball/model/best_2.rknn \
  --labels /home/elf/work/basketball/model/labels.txt \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
  --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
  --width 1920 --height 1080 --fps 30 \
  --conf 0.25 --nms 0.45 \
  --rknn-core -1 \
  --runtime-seam-x 150 \
  --runtime-blend-width 40 \
  --runtime-right-x-shift 30 \
  --runtime-right-y-shift -5 \
  --detect-interval 1 \
  --smooth 0.65 \
  --display-scale 0.25
```

### 6.11 自动运镜 1920x1080 输出

```bash
cd /home/elf/work/basketball/camera_movement

python3 dual_rknn_director_view1920_fast.py \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --model /home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn \
  --labels /home/elf/work/basketball/model/labels.txt \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
  --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params.npz \
  --width 1920 --height 1080 --fps 30 \
  --conf 0.25 --nms 0.45 \
  --rknn-core -1 \
  --runtime-seam-x 150 \
  --runtime-blend-width 40 \
  --runtime-right-x-shift 30 \
  --runtime-right-y-shift 0 \
  --detect-interval 3 \
  --smooth 0.70 \
  --view-width 1920 \
  --view-height 1080 \
  --crop-y-mode center \
  --display-scale 0.5 \
  --print-every 30
```

建议：若目标是最终输出 20 FPS 以上，`--detect-interval 3` 或 `4` 通常更现实；RKNN 检测低频更新，导播视口高频输出。

---

## 7. 参数文件说明

| 文件 | 生成脚本 | 用途 |
|---|---|---|
| `camera_usb2_calib.npz` | `calib_usb2_camera.py --mode calibrate` | 左相机内参、畸变系数、new camera matrix、重投影误差。 |
| `camera_calib.npz` | `calib_usb_camera.py --mode calibrate` | 右相机内参、畸变系数、new camera matrix、重投影误差。 |
| `stereo_rectify_maps_wide.npz` | `offline_build_stereo_rectify_maps.py --mode build` | 双目 raw -> rectified remap、R/T、R1/R2/P1/P2/Q、roi 等。 |
| `stereo_rectify_maps_wide_good.npz` | 人工复制 | 确认效果好的稳定 remap。 |
| `stitch_params.npz` | `offline_estimate_stitch_params.py` | overlap、vertical offset、ROI、alpha mask、输出尺寸等。 |
| `stitch_params_good.npz` | 人工复制或手动估计输出 | 确认效果好的稳定拼接参数。 |

实时脚本通常需要同时指定：

```text
--map-file      stereo_rectify_maps*_good.npz
--stitch-param  stitch_params*_good.npz
```

---

## 8. 常见问题与处理

### 8.1 图像重复 remap

现象：拼接参数离谱、画面变形、接不上。

原因：输入已经是 rectified 图，但又被当作 raw 图 remap 了一次。

处理：运行 `offline_estimate_stitch_params.py` 时，如果输入是 `left_rect/right_rect` 或 `finish_stereo_rectify` 中的图，添加：

```bash
--input-already-rectified
```

### 8.2 极线不水平或右图翻转

优先检查：

1. 左右双目标定图是否按相同编号配对。
2. 棋盘格是否在左右图中都清晰可见。
3. `--right-corner-transform` 是否正确。

可尝试：

```text
--right-corner-transform orig
--right-corner-transform flip_ud
--right-corner-transform flip_lr
--right-corner-transform rot180
```

### 8.3 拼接接缝明显

优先调整：

```text
overlap_px
vertical_offset
blend_width
runtime-seam-x
runtime-right-x-shift
runtime-right-y-shift
```

经验：

| 现象 | 调整方向 |
|---|---|
| 左右画面横向接不上 | 调 `overlap_px` 或 `runtime-right-x-shift` |
| 左右画面上下错位 | 调 `vertical_offset` 或 `runtime-right-y-shift` |
| 接缝生硬 | 增大 `blend_width` |
| 人物重影明显 | 减小 `blend_width`，避免把接缝放在近距离运动人物上 |

### 8.4 帧率低

优先：

```bash
--disable-color-balance
--display-scale 0.10
--threaded-capture
```

避免开启：

```text
--show-rectified
--show-lines
--save-every
```

说明：

- 完整宽幅分辨率较大，显示和颜色补偿都会明显消耗 CPU / 内存带宽。
- RKNN 单次推理约几十毫秒时，不建议每帧检测；自动运镜脚本使用 `--detect-interval` 降低检测频率。
- 如果只是最终节目输出，优先使用 `realtime_stereo_stitch_1920_view.py` 或 `dual_rknn_director_view1920_fast.py`，不要每帧生成完整宽幅图。

### 8.5 摄像头打不开或掉帧

检查：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video41 --list-formats-ext
v4l2-ctl -d /dev/video43 --list-formats-ext
```

建议：

- 默认使用 MJPG，降低 USB 带宽压力。
- 如果指定 `--no-mjpg` 后出现卡顿或打不开，去掉该参数。
- 标定、双目标定、实时运行必须尽量使用同一分辨率。

### 8.6 OpenCV 窗口无法显示

如果没有 `DISPLAY`：

```bash
export DISPLAY=:0
```

或使用：

```bash
--headless
```

多数脚本同时支持终端输入按键，例如在终端输入 `s` 后回车保存。

---

## 9. 当前推荐保留的主流程文件

主流程建议保留并优先维护：

```text
calib_usb2_camera.py
calib_usb_camera.py
offline_build_stereo_rectify_maps.py
offline_estimate_stitch_params.py
realtime_stereo_stitch.py
realtime_stereo_stitch_1920_view.py
infer_person.py
camera_movement/dual_rknn_wide_coords_display_alternating.py
camera_movement/dual_rknn_director_view1920_fast.py
camera_movement/dual_rknn_director_view1920_record_local.py
camera_movement/predict1_weighted.py
camera_movement/predict1_director.py
save_wide_stitch_photo_to_tf.py
save_wide_and_view_photo_to_tf.py
record_select_h264_rkmpp_video.py
record_select_h264_rkmpp_rectified_option.py
```

稳定参数建议保留：

```text
camera_usb2_calib.npz
camera_calib.npz
offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz
offline_build_stereo_rectify_maps/stitch_params_good.npz
model/
```

可按需清理的调试输出：

```bash
rm -rf realtime_stitch_debug
rm -rf stereo_rectify_debug
rm -rf offline_build_stereo_rectify_maps/stitch_param_debug
```

---

## 10. 最短启动清单

如果已有稳定标定和拼接参数，只想启动最终自动运镜：

```bash
cd /home/elf/work/basketball/camera_movement

python3 dual_rknn_director_view1920_fast.py \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --model /home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn \
  --labels /home/elf/work/basketball/model/labels.txt \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
  --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
  --width 1920 --height 1080 --fps 30 \
  --conf 0.25 --nms 0.45 \
  --detect-interval 3 \
  --view-width 1920 --view-height 1080 \
  --display-scale 0.5
```

如果只想验证拼接画面：

```bash
cd /home/elf/work/basketball

python3 realtime_stereo_stitch_1920_view.py \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
  --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
  --width 1920 --height 1080 --fps 30 \
  --view-width 1920 --view-height 1080 \
  --display-scale 0.25 \
  --disable-color-balance \
  --threaded-capture
```
