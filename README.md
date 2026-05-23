# Basketball Match 双目拼接与球员检测项目

本项目用于在 RK3588 / Linux 平台上实现 **双 USB 摄像头实时篮球场宽视角拼接**，并为后续 **RKNN 球员检测** 提供统一的宽画面输入。

当前项目已经完成以下核心流程：

```text
左 / 右 USB 摄像头
    ↓
单目标定
    ↓
离线生成双目 remap 查找表
    ↓
离线估计 overlap / vertical_offset / alpha mask
    ↓
实时双目 remap
    ↓
裁剪 / 平移 / alpha 融合
    ↓
输出实时拼接图
    ↓
RKNN 球员检测
```

---

## 1. 项目目标

篮球比赛场景中，单个摄像头通常无法覆盖完整篮球场，尤其是在固定机位、低成本 USB 摄像头和边缘计算平台条件下，单摄视野有限。

本项目通过两个固定摄像头分别拍摄篮球场左右区域，再通过离线标定和实时融合，生成一个更宽的拼接画面。该画面后续可以送入 RKNN 模型进行球员检测、ROI 分析、人数统计或报警处理。

项目主要目标：

1. 使用两路 USB 摄像头采集篮球场画面。
2. 对每个摄像头进行单目标定，减小镜头畸变影响。
3. 离线生成双目几何校正 remap 查找表。
4. 离线估计左右图像的重叠区域和上下偏移。
5. 实时完成双路图像矫正、裁剪、融合和显示。
6. 保留 RKNN 推理代码，方便后续接入球员检测。
7. 为后续 20 FPS 以上实时处理做性能优化基础。

---

## 2. 当前硬件配置

当前使用两路 USB 摄像头：

| 摄像头 | 设备节点 | 说明 |
|---|---|---|
| 左摄像头 | `/dev/video41` | USB2 摄像头 |
| 右摄像头 | `/dev/video43` | USB 摄像头 |

默认采集参数：

```text
分辨率：1920 x 1080
帧率：30 FPS
格式：MJPG
平台：RK3588 / Linux
```

查看摄像头设备：

```bash
v4l2-ctl --list-devices
```

查看摄像头支持格式：

```bash
v4l2-ctl -d /dev/video41 --list-formats-ext
v4l2-ctl -d /dev/video43 --list-formats-ext
```

查看曝光、白平衡、增益控制项：

```bash
v4l2-ctl -d /dev/video41 --list-ctrls-menus
v4l2-ctl -d /dev/video43 --list-ctrls-menus
```

---

## 3. 推荐保留的核心文件

当前最终主线建议只保留以下 Python 脚本：

```text
calib_usb_camera.py
calib_usb2_camera.py
offline_build_stereo_rectify_maps.py
offline_estimate_stitch_params.py
realtime_stereo_stitch.py
infer_person.py
```

文件作用如下：

| 文件 | 作用 |
|---|---|
| `calib_usb2_camera.py` | 左摄像头单目标定，生成 `camera_usb2_calib.npz` |
| `calib_usb_camera.py` | 右摄像头单目标定，生成 `camera_calib.npz` |
| `offline_build_stereo_rectify_maps.py` | 离线生成双目极线校正 remap 查找表 |
| `offline_estimate_stitch_params.py` | 离线估计 `overlap_px / vertical_offset / alpha_mask` |
| `realtime_stereo_stitch.py` | 实时读取双摄、remap、裁剪、融合、显示 |
| `infer_person.py` | RKNN 球员检测推理脚本 |

建议删除或归档的旧脚本包括：

```text
align_cameras.py
stereo_calibrate_from_undistorted_images.py
realtime_rectify_preview.py
stereo_rectify_stitch_live.py
stereo_rectify_wide_only.py
calib_imx219_camera.py
calib_ov13588_camera.py
*.bak*
__pycache__/
```

这些文件属于早期尝试、旧路线或备份版本，不建议继续放在根目录中，避免误用旧代码和旧参数。

---

## 4. 推荐目录结构

建议整理后的项目目录如下：

```text
basketball/
├── README.md
│
├── calib_usb_camera.py
├── calib_usb2_camera.py
│
├── offline_build_stereo_rectify_maps.py
├── offline_estimate_stitch_params.py
├── realtime_stereo_stitch.py
├── infer_person.py
│
├── camera_calib.npz
├── camera_usb2_calib.npz
│
├── offline_build_stereo_rectify_maps/
│   ├── stereo_rectify_maps_wide_good.npz
│   ├── stitch_params_good.npz
│   ├── stereo_rectify_maps_1280x720.npz
│   ├── stitch_params_1280x720.npz
│
├── model/
│   ├── labels.txt
│   ├── basketball_player.onnx
│   ├── basketball_player.rknn
│   ├── basketball_player_2.1.0.rknn
│   ├── basketball_player_fp_2.1.0.rknn
│   └── best_2.rknn
│
└── .gitignore
```

调试输出目录建议不要提交到 GitHub：

```text
realtime_stitch_debug/
stereo_rectify_debug/
stereo_rectify_live_debug/
stereo_stitch_debug/
offline_build_stereo_rectify_maps/stitch_param_debug/
offline_build_stereo_rectify_maps/stitch_param_debug_1280x720/
```

---

## 5. 环境依赖

### 5.1 Python 依赖

```bash
python3 -m pip install numpy opencv-python
```

如果系统 OpenCV 是手动编译安装，可以检查 Python 是否能正常导入：

```bash
python3 - <<'PY'
import cv2
import numpy as np
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
PY
```

### 5.2 系统工具

需要安装 `v4l2-ctl`：

```bash
sudo apt install v4l-utils
```

### 5.3 RKNN 推理依赖

如果使用 `infer_person.py` 进行 RKNN 推理，需要安装 RKNN Runtime / Toolkit Lite2，具体版本取决于当前 RK3588 系统环境。

---

## 6. 整体技术路线

当前最终流程分为四个阶段：

```text
阶段一：单目标定
    calib_usb2_camera.py
    calib_usb_camera.py

阶段二：离线生成双目 remap
    offline_build_stereo_rectify_maps.py

阶段三：离线估计拼接参数
    offline_estimate_stitch_params.py

阶段四：实时双目拼接
    realtime_stereo_stitch.py

可选阶段五：RKNN 球员检测
    infer_person.py
```

---

## 7. 单目标定流程

如果项目中已经存在以下两个文件，可以跳过本节：

```text
camera_usb2_calib.npz
camera_calib.npz
```

### 7.1 左摄像头单目标定

```bash
python3 calib_usb2_camera.py --mode capture
python3 calib_usb2_camera.py --mode calibrate
```

生成：

```text
camera_usb2_calib.npz
```

### 7.2 右摄像头单目标定

```bash
python3 calib_usb_camera.py --mode capture
python3 calib_usb_camera.py --mode calibrate
```

生成：

```text
camera_calib.npz
```

### 7.3 单目标定图片要求

建议采集时注意：

```text
1. 棋盘格完整、清晰。
2. 不要严重反光。
3. 不要运动模糊。
4. 棋盘格不要贴边。
5. 覆盖画面左上、右上、左下、右下和中间。
6. 距离和角度要有变化。
7. 每个摄像头建议采集 30~60 张。
8. 单目标定、双目标定和实时运行尽量使用同一分辨率。
```

---

## 8. 离线生成双目 remap 查找表

脚本：

```text
offline_build_stereo_rectify_maps.py
```

该脚本负责：

```text
读取左右单目标定参数
读取双目标定图片
检测棋盘格角点
估计双目外参
生成 raw -> rectified 的 remap 查找表
保存 stereo_rectify_maps_xxx.npz
```

### 8.1 高分辨率 remap 生成

```bash
python3 offline_build_stereo_rectify_maps.py \
    --mode build \
    --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
    --right-calib-file /home/elf/work/basketball/camera_calib.npz \
    --capture-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/images \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
    --board-cols 11 \
    --board-rows 8 \
    --square-size 22 \
    --mono-dist-scale 0.8 \
    --rectify-alpha 1.0 \
    --out-scale 1.8 \
    --right-corner-transform orig \
    --headless
```

生成：

```text
offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz
```

如果效果确认可用，建议备份为稳定版本：

```bash
cp offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
   offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz
```

### 8.2 1280x720 加速版 remap

如果目标是 20 FPS 以上，建议使用较小输出画布：

```bash
python3 offline_build_stereo_rectify_maps.py \
    --mode build \
    --left-calib-file /home/elf/work/basketball/camera_usb2_calib.npz \
    --right-calib-file /home/elf/work/basketball/camera_calib.npz \
    --capture-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/images \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_1280x720.npz \
    --board-cols 11 \
    --board-rows 8 \
    --square-size 22 \
    --mono-dist-scale 0.8 \
    --rectify-alpha 1.0 \
    --out-width 1280 \
    --out-height 720 \
    --right-corner-transform orig \
    --headless
```

生成：

```text
offline_build_stereo_rectify_maps/stereo_rectify_maps_1280x720.npz
```

### 8.3 参数说明

| 参数 | 说明 |
|---|---|
| `--mono-dist-scale 0.8` | 单目畸变矫正强度。当前项目中 1.0 会有过矫正，因此使用 0.8 |
| `--rectify-alpha 1.0` | 尽量保留视野，黑边可能更多 |
| `--out-scale` | 放大 rectified 输出画布，减少裁切 |
| `--out-width / --out-height` | 手动指定 rectified 输出尺寸 |
| `--right-corner-transform orig` | 使用原始右图角点顺序。多种翻转模式测试后，当前以 `orig` 为准 |

---

## 9. 离线估计拼接参数

脚本：

```text
offline_estimate_stitch_params.py
```

该脚本负责：

```text
读取左右图
读取 remap 文件
生成 rectified 图
转灰度
估计 overlap_px
估计 vertical_offset
生成 alpha 融合 mask
保存 stitch_params.npz
```

### 9.1 输入图片类型说明

该脚本支持三种输入：

| 输入类型 | 参数 |
|---|---|
| 摄像头 raw 原图 | 不加额外输入类型参数 |
| 已单目矫正图 | `--input-already-undistorted` |
| 已极线矫正图 | `--input-already-rectified` |

如果输入的是类似：

```text
live_000000_left_rect.jpg
live_000000_right_rect.jpg
```

这种已经 rectified 的图，必须加：

```bash
--input-already-rectified
```

否则脚本会重复 remap，导致拼接参数错误。

---

### 9.2 自动估计高分辨率拼接参数

当前效果较好的参数为：

```text
overlap_px      = 1010
vertical_offset = 41
blend_width     = 60
```

自动估计示例：

```bash
python3 offline_estimate_stitch_params.py \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
    --left-image /home/elf/work/basketball/stereo_rectify_live_debug/live_000000_left_rect.jpg \
    --right-image /home/elf/work/basketball/stereo_rectify_live_debug/live_000000_right_rect.jpg \
    --output-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params.npz \
    --output-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_param_debug \
    --input-already-rectified \
    --min-overlap 700 \
    --max-overlap 1200 \
    --max-vertical-offset 100 \
    --search-scale 0.35
```

如果自动结果已经很好，建议固定参数，不要频繁重新估计：

```bash
cp offline_build_stereo_rectify_maps/stitch_params.npz \
   offline_build_stereo_rectify_maps/stitch_params_good.npz
```

---

### 9.3 手动指定拼接参数

如果已经知道好用的参数，可以手动指定：

```bash
python3 offline_estimate_stitch_params.py \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide.npz \
    --left-image /home/elf/work/basketball/stereo_rectify_live_debug/live_000000_left_rect.jpg \
    --right-image /home/elf/work/basketball/stereo_rectify_live_debug/live_000000_right_rect.jpg \
    --output-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
    --output-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_param_debug \
    --input-already-rectified \
    --manual-overlap 1010 \
    --manual-vertical-offset 41 \
    --blend-width 60
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--manual-overlap` | 左右图重叠宽度 |
| `--manual-vertical-offset` | 右图相对左图的上下偏移 |
| `--blend-width` | 真正 alpha 渐变融合区域宽度 |

经验：

```text
接不上：调整 overlap_px
上下错位：调整 vertical_offset
接缝硬：增大 blend_width
重影明显：减小 blend_width
```

---

### 9.4 1280x720 拼接参数

如果使用 `stereo_rectify_maps_1280x720.npz`，需要生成对应的：

```text
stitch_params_1280x720.npz
```

示例：

```bash
python3 offline_estimate_stitch_params.py \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_1280x720.npz \
    --left-image /home/elf/work/basketball/realtime_stitch_debug/frame_000000_left_raw.jpg \
    --right-image /home/elf/work/basketball/realtime_stitch_debug/frame_000000_right_raw.jpg \
    --output-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_1280x720.npz \
    --output-dir /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_param_debug_1280x720 \
    --manual-overlap 586 \
    --manual-vertical-offset 24 \
    --blend-width 50
```

注意：这里输入的是摄像头 raw 图，因此不要加：

```text
--input-already-rectified
--input-already-undistorted
```

---

## 10. 实时双目拼接

脚本：

```text
realtime_stereo_stitch.py
```

作用：

```text
读取左右摄像头 raw 图
使用 stereo_rectify_maps_xxx.npz 做 remap
使用 stitch_params_xxx.npz 做裁剪、平移和融合
实时显示 stitched 结果
可保存 raw / rectified / stitched 调试图
```

### 10.1 高分辨率实时拼接

```bash
python3 realtime_stereo_stitch.py \
    --left-device /dev/video41 \
    --right-device /dev/video43 \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
    --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --display-scale 0.15 \
    --disable-color-balance
```

### 10.2 1280x720 加速版实时拼接

```bash
python3 realtime_stereo_stitch.py \
    --left-device /dev/video41 \
    --right-device /dev/video43 \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_1280x720.npz \
    --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_1280x720.npz \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --display-scale 0.25 \
    --disable-color-balance
```

---

## 11. 实时脚本按键

运行 `realtime_stereo_stitch.py` 后支持：

| 按键 | 功能 |
|---|---|
| `q` / `Esc` | 退出 |
| `s` | 保存当前帧调试图 |
| `b` | 开启 / 关闭颜色补偿 |
| `l` | 开启 / 关闭水平参考线 |
| `r` | 开启 / 关闭左右 rectified 调试窗口 |

如果 OpenCV 窗口没有焦点，也可以在终端输入：

```text
s 回车
```

保存调试图。

保存目录：

```text
realtime_stitch_debug/
```

保存内容包括：

```text
frame_xxxxxx_left_raw.jpg
frame_xxxxxx_right_raw.jpg
frame_xxxxxx_left_rect.jpg
frame_xxxxxx_right_rect.jpg
frame_xxxxxx_stitched.jpg
```

---

## 12. 帧率优化建议

当前高分辨率版本处理量较大：

```text
rectified_size: 2208 x 1242
stitched_size : 约 3406 x 1201
```

如果目标是 20 FPS 以上，建议：

```text
1. 使用 1280x720 remap
2. 关闭颜色补偿：--disable-color-balance
3. 降低显示比例：--display-scale 0.10 ~ 0.25
4. 不开启 --show-rectified
5. 不开启 --show-lines
6. 不开启 --save-every
7. 后续可以进一步做 ROI remap 优化
```

### 12.1 颜色补偿

颜色补偿会对图像做大面积浮点运算，明显降低 FPS。

如果更关注实时性能，建议关闭：

```bash
--disable-color-balance
```

如果更关注接缝颜色一致，可以开启，但帧率会下降。

### 12.2 MJPG 与 YUYV

当前默认使用 MJPG，可以降低 USB 带宽压力，但 CPU 需要解码 JPEG。

如果 USB 带宽足够，可以尝试关闭 MJPG：

```bash
--no-mjpg
```

如果出现掉帧或摄像头打不开，则继续使用 MJPG。

---

## 13. RKNN 球员检测

项目保留 RKNN 推理代码：

```text
infer_person.py
```

模型目录：

```text
model/
├── labels.txt
├── basketball_player.onnx
├── basketball_player.rknn
├── basketball_player_2.1.0.rknn
├── basketball_player_fp_2.1.0.rknn
└── best_2.rknn
```

后续推荐流程：

```text
realtime_stereo_stitch.py 输出 stitched frame
    ↓
resize 到模型输入尺寸
    ↓
infer_person.py / RKNN 推理
    ↓
绘制检测框
```

---

## 14. 常见问题

### 14.1 接缝明显

优先检查：

```text
overlap_px
vertical_offset
blend_width
```

处理建议：

```text
画面接不上：调整 overlap_px
上下错位：调整 vertical_offset
接缝太硬：增大 blend_width
重影严重：减小 blend_width
```

### 14.2 拼接有重影

可能原因：

```text
1. blend_width 太大
2. 左右摄像头不同步
3. 场景中近距离物体视差大
4. overlap_px 不准确
5. 摄像头位置发生变化
```

建议：

```text
1. 固定好效果最好的 stitch_params_good.npz
2. 不要频繁重新估计参数
3. 只微调 blend_width
4. 尽量避免把接缝放在近距离运动物体上
```

### 14.3 保存不了图片

实时脚本保存依赖按键。

方法一：点击 OpenCV 图像窗口，然后按：

```text
s
```

方法二：在终端输入：

```text
s 回车
```

保存目录：

```text
realtime_stitch_debug/
```

### 14.4 帧率低

优先使用：

```bash
--disable-color-balance
--display-scale 0.10
```

不要使用：

```bash
--show-rectified
--show-lines
--save-every
```

如果仍然低，使用：

```text
stereo_rectify_maps_1280x720.npz
stitch_params_1280x720.npz
```

### 14.5 图像重复 remap

如果输入已经是 rectified 图，例如：

```text
live_000000_left_rect.jpg
live_000000_right_rect.jpg
```

在 `offline_estimate_stitch_params.py` 中必须加：

```bash
--input-already-rectified
```

否则脚本会把它当成 raw 图继续 remap，导致参数错误。

---

## 15. 本地清理建议

如果只想删除本地旧文件，不想影响 GitHub 仓库，请使用 `rm`，不要使用 `git rm`。

推荐删除：

```bash
rm -f align_cameras.py
rm -f stereo_calibrate_from_undistorted_images.py
rm -f realtime_rectify_preview.py
rm -f stereo_rectify_stitch_live.py
rm -f stereo_rectify_wide_only.py
rm -f calib_imx219_camera.py
rm -f calib_ov13588_camera.py

rm -f *.bak
rm -f *.bak_*
rm -rf __pycache__

rm -rf realtime_stitch_debug
rm -rf stereo_rectify_debug
rm -rf stereo_rectify_live_debug
rm -rf stereo_stitch_debug
rm -rf stereo_undistorted_calib
```

推荐保留：

```text
calib_usb_camera.py
calib_usb2_camera.py
offline_build_stereo_rectify_maps.py
offline_estimate_stitch_params.py
realtime_stereo_stitch.py
infer_person.py
camera_calib.npz
camera_usb2_calib.npz
offline_build_stereo_rectify_maps/*.npz
model/
```

---

## 16. 推荐最终运行流程

### 第一步：确认最终 remap 和 stitch 参数

高分辨率：

```text
offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz
offline_build_stereo_rectify_maps/stitch_params_good.npz
```

加速版本：

```text
offline_build_stereo_rectify_maps/stereo_rectify_maps_1280x720.npz
offline_build_stereo_rectify_maps/stitch_params_1280x720.npz
```

### 第二步：启动实时拼接

高分辨率：

```bash
python3 realtime_stereo_stitch.py \
    --left-device /dev/video41 \
    --right-device /dev/video43 \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz \
    --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --display-scale 0.15 \
    --disable-color-balance
```

加速版：

```bash
python3 realtime_stereo_stitch.py \
    --left-device /dev/video41 \
    --right-device /dev/video43 \
    --map-file /home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_1280x720.npz \
    --stitch-param /home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_1280x720.npz \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --display-scale 0.25 \
    --disable-color-balance
```

---

## 17. 当前最终主线

```text
calib_usb2_camera.py
calib_usb_camera.py
        ↓
offline_build_stereo_rectify_maps.py
        ↓
offline_estimate_stitch_params.py
        ↓
realtime_stereo_stitch.py
        ↓
infer_person.py
```

当前项目已经完成：

```text
1. USB 摄像头单目标定
2. 双目 remap 查找表生成
3. 拼接参数估计
4. 实时双目拼接
5. RKNN 推理代码保留
```

后续可继续优化：

```text
1. ROI remap 提速
2. C++ / RGA 硬件加速
3. 实时 RKNN 检测接入
4. 篮球场 ROI 区域分析
5. 人员检测、计数、报警
```
