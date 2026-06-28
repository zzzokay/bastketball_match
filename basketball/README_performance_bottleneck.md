# Basketball 性能瓶颈验证日志

日期：2026-06-27

本日志记录本次对 `/home/elf/work/basketball` 项目现有实时链路的性能瓶颈验证结果，用于后续优化前的基线参考。

本次验证目的不是改代码、不是优化，而是先确认当前链路到底卡在哪：

```text
双路采集 → MJPG 解码 → remap 矫正 → 拼接融合 → 显示 → RKNN → camera_movement
```

重点记录：

```text
FPS
延迟 / 单帧耗时
CPU 占用
瓶颈判断
推荐程度
```

---

## 1. 重要说明

本次验证遵守以下原则：

1. **没有修改项目现有脚本**。
2. 没有修改：

```text
calib_usb_camera.py
calib_usb2_camera.py
offline_build_stereo_rectify_maps.py
offline_estimate_stitch_params.py
realtime_stereo_stitch.py
realtime_stereo_stitch_1920_view.py
camera_movement/*.py
```

3. 临时测试脚本只放在 `/tmp` 下：

```text
/tmp/basketball_bottleneck_benchmark.py
/tmp/basketball_display_window_benchmark.py
```

4. 临时测试结果输出到：

```text
/tmp/basketball_bottleneck_result.json
/tmp/basketball_bottleneck_display_result.json
/tmp/basketball_display_window_result.json
/tmp/cm_director_profile.log
/tmp/cm_director_cpu_samples.txt
/tmp/cm_wide_profile.log
/tmp/cm_wide_cpu_samples.txt
```

5. 现有项目文件只被读取，不被改动。

---

## 2. 测试环境

| 项目 | 值 |
|---|---|
| 平台 | RK3588 / ELF2 / Linux |
| 左摄像头 | `/dev/video41` |
| 右摄像头 | `/dev/video43` |
| 请求分辨率 | `1920 x 1080` |
| 请求帧率 | `30 FPS` |
| 摄像头格式 | `MJPG` |
| OpenCV | `4.10.0` |
| NumPy | `2.2.1` |
| RKNNLite | 可导入，运行正常 |
| 显示环境 | 初始无 `DISPLAY`，后续按要求使用 `DISPLAY=:0` |

摄像头实际打开结果：

| 设备 | 实际分辨率 | 实际 FPS | FOURCC |
|---|---:|---:|---|
| `/dev/video41` | `1920 x 1080` | `30.0` | `MJPG` |
| `/dev/video43` | `1920 x 1080` | `30.0` | `MJPG` |

使用的参数文件：

```text
/home/elf/work/basketball/offline_build_stereo_rectify_maps/stereo_rectify_maps_wide_good.npz
/home/elf/work/basketball/offline_build_stereo_rectify_maps/stitch_params_good.npz
```

读取到的关键参数：

```text
raw_image_size : 1920 x 1080
rectified_size : 2208 x 1242
overlap_px     : 990
vertical_offset: -20
blend_width    : 60
output_size    : 3426 x 1222
```

---

## 3. 本次测试矩阵

按照“技术路线”中要求，验证以下链路：

| 编号 | 测试项 | 目的 |
|---|---|---|
| T1 | 双路采集 grab only | 判断摄像头同步抓帧等待情况 |
| T2 | 双路采集 + MJPG 解码 | 判断 USB 采集和 MJPG 解码开销 |
| T3 | 双路采集 + remap | 判断读取矫正参数后每帧矫正开销 |
| T4 | 完整范围拼接 | 判断完整宽幅 remap + 拼接融合开销 |
| T5 | 完整拼接 + 显示前 resize | 判断显示前缩放开销 |
| T6 | `DISPLAY=:0` 完整拼接 + LCD 显示 | 判断真实 LCD 显示链路影响 |
| T7 | 完整拼接 + RKNN | 判断宽图检测调试路线性能 |
| T8 | 当前 camera_movement 主路线 | 判断最终自动运镜路线性能 |

---

## 4. 临时 benchmark 测试内容

### 4.1 分阶段 benchmark 脚本

临时脚本：

```text
/tmp/basketball_bottleneck_benchmark.py
```

作用：

```text
只读取摄像头和现有 npz 参数
不修改项目文件
按场景逐步叠加测试：
  grab
  decode
  remap
  stitch
  display_prep
```

运行命令：

```bash
python3 /tmp/basketball_bottleneck_benchmark.py \
  --seconds 8 \
  --warmup 20 \
  --scenarios grab,decode,remap,stitch,display_prep \
  --output /tmp/basketball_bottleneck_result.json
```

### 4.2 DISPLAY=:0 下显示前处理测试

运行命令：

```bash
DISPLAY=:0 python3 /tmp/basketball_bottleneck_benchmark.py \
  --seconds 8 \
  --warmup 20 \
  --scenarios display_prep \
  --output /tmp/basketball_bottleneck_display_result.json
```

说明：这个测试只测完整拼接后 resize 的显示前处理，不调用真实 `cv2.imshow()`。

### 4.3 DISPLAY=:0 下真实 OpenCV 显示测试

临时脚本：

```text
/tmp/basketball_display_window_benchmark.py
```

作用：

```text
完整执行：
双路采集 + MJPG 解码 + remap + 拼接融合 + resize + cv2.imshow + cv2.waitKey(1)
```

运行命令：

```bash
DISPLAY=:0 python3 /tmp/basketball_display_window_benchmark.py \
  --seconds 8 \
  --display-scale 0.25 \
  --output /tmp/basketball_display_window_result.json
```

---

## 5. 分阶段测试结果

### 5.1 T1：双路 grab only

结果：

| 指标 | 数值 |
|---|---:|
| FPS | `10.08` |
| 总延迟 / 帧 | `99.18 ms` |
| grab 平均耗时 | `99.16 ms` |
| 系统 CPU | `2.0%` |
| 进程 CPU | `0.2%` 单核占比 |

判断：

```text
grab only 只抓帧不解码，不完全代表真实业务链路。
这个结果更像是在等待摄像头帧同步。
不把它作为最终瓶颈判断依据，只作为参考。
```

推荐程度：

```text
仅参考
```

---

### 5.2 T2：双路采集 + MJPG 解码

结果：

| 指标 | 数值 |
|---|---:|
| FPS | `15.01` |
| 总延迟 / 帧 | `66.61 ms` |
| grab 平均耗时 | `40.90 ms` |
| retrieve / MJPG 解码平均耗时 | `25.16 ms` |
| 系统 CPU | `7.8%` |
| 进程 CPU | `38.7%` 单核占比 |

判断：

```text
MJPG 解码已经有明显开销。
双路 1920x1080 MJPG 解码约 21~25 ms / 帧对。
这一步已经占掉 30FPS 单帧预算 33.3ms 的大半。
```

推荐程度：

```text
中：基础采集链路可用，但 MJPG 解码是明确开销点。
```

---

### 5.3 T3：双路采集 + remap

结果：

| 指标 | 数值 |
|---|---:|
| FPS | `14.96` |
| 总延迟 / 帧 | `66.83 ms` |
| grab 平均耗时 | `12.70 ms` |
| retrieve / MJPG 解码平均耗时 | `21.65 ms` |
| 双路 remap 平均耗时 | `32.45 ms` |
| 系统 CPU | `49.9%` |
| 进程 CPU | `382.4%` 单核占比，约 3.8 核 |

判断：

```text
remap 是主要瓶颈之一。
即使矫正参数已经提前生成，每一帧仍然要对输出图每个像素查表、插值和写回。
当前 rectified_size 为 2208 x 1242。
双路 remap 每帧约处理 548 万输出像素。
```

关键结论：

```text
读取矫正参数本身不耗 CPU。
真正耗 CPU 的是每帧 cv2.remap。
```

推荐程度：

```text
中：作为必要基础步骤可以接受，但不适合每帧做完整双路 full-frame remap 后再继续做完整宽图。
```

---

### 5.4 T4：完整范围拼接

结果：

| 指标 | 数值 |
|---|---:|
| FPS | `10.30` |
| 总延迟 / 帧 | `97.09 ms` |
| retrieve / MJPG 解码平均耗时 | `21.70 ms` |
| 双路 remap 平均耗时 | `32.39 ms` |
| stitch / 拼接融合平均耗时 | `42.40 ms` |
| 系统 CPU | `40.6%` |
| 进程 CPU | `311.3%` 单核占比，约 3.1 核 |

判断：

```text
完整宽幅拼接融合是最大常规瓶颈之一。
完整拼接输出尺寸约 3426 x 1222。
完整宽图拼接融合约 40~42 ms / 帧。
加上 MJPG 解码和 remap 后，总耗时接近 100 ms，只能达到约 10 FPS。
```

推荐程度：

```text
低：适合调试，不适合最终实时输出。
```

---

### 5.5 T5：完整拼接 + 显示前 resize

结果：

| 指标 | 数值 |
|---|---:|
| FPS | `10.06` |
| 总延迟 / 帧 | `99.43 ms` |
| retrieve / MJPG 解码平均耗时 | `21.77 ms` |
| remap 平均耗时 | `32.14 ms` |
| stitch 平均耗时 | `41.52 ms` |
| resize 显示前处理平均耗时 | `3.38 ms` |
| 系统 CPU | `41.9%` |
| 进程 CPU | `315.2%` 单核占比，约 3.15 核 |

判断：

```text
显示前 resize 本身不是最大瓶颈。
完整宽图链路慢，主要还是 remap + stitch。
```

推荐程度：

```text
低：完整宽图显示前处理可用于调试，不建议作为最终路线。
```

---

## 6. DISPLAY=:0 显示链路测试结果

用户允许使用：

```bash
export DISPLAY=:0
```

本次实际在命令前临时加：

```bash
DISPLAY=:0
```

---

### 6.1 DISPLAY=:0 下完整拼接 + resize

结果：

| 指标 | 数值 |
|---|---:|
| FPS | `10.20` |
| 总延迟 / 帧 | `98.01 ms` |
| retrieve / MJPG 解码平均耗时 | `21.79 ms` |
| remap 平均耗时 | `32.33 ms` |
| stitch 平均耗时 | `39.99 ms` |
| resize 平均耗时 | `3.27 ms` |
| 系统 CPU | `41.6%` |
| 进程 CPU | `317.8%` 单核占比 |

判断：

```text
只是设置 DISPLAY，但不真正 imshow 时，结果和 headless 基本一致。
```

---

### 6.2 DISPLAY=:0 下完整拼接 + 真实 cv2.imshow

结果：

| 指标 | 数值 |
|---|---:|
| FPS | `3.79` |
| 总延迟 / 帧 | `263.85 ms` |
| retrieve / MJPG 解码平均耗时 | `21.86 ms` |
| remap 平均耗时 | `191.64 ms` |
| stitch 平均耗时 | `38.63 ms` |
| resize 平均耗时 | `8.75 ms` |
| imshow + waitKey 平均耗时 | `2.93 ms` |
| 系统 CPU | `25.7%` |
| 进程 CPU | `184.9%` 单核占比 |

判断：

```text
cv2.imshow/waitKey 自身平均约 2.9 ms，不是单独最大耗时。
但真实 DISPLAY=:0 图形环境下，完整宽图显示链路整体明显变慢。
本轮观测中 remap 在 GUI 显示条件下被拉高到约 191 ms。
因此完整宽图 + LCD 显示不适合作为实时主链路。
```

推荐程度：

```text
很低：只适合临时观察，不建议最终运行时显示完整宽图。
```

---

## 7. 完整拼接 + RKNN 测试

测试脚本：

```text
/home/elf/work/basketball/camera_movement/dual_rknn_wide_coords_display_alternating.py
```

运行目的：

```text
验证完整宽幅图 + RKNN 检测 + 坐标映射 + 显示的综合性能。
```

运行命令：

```bash
DISPLAY=:0 timeout 15s python3 \
  /home/elf/work/basketball/camera_movement/dual_rknn_wide_coords_display_alternating.py \
  --left-device /dev/video41 \
  --right-device /dev/video43 \
  --model /home/elf/work/basketball/model/best_2.rknn \
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
  --detect-interval 1 \
  --smooth 0.65 \
  --display-scale 0.25 \
  --print-every 10 \
  --save-dir /tmp/basketball_debug_wide
```

结果摘要：

| 指标 | 数值 |
|---|---:|
| FPS | `7.0 ~ 7.3` |
| total 单帧耗时 | `约 168 ~ 181 ms` |
| RKNN infer | `约 52 ~ 57 ms` |
| CPU 采样 | 稳定后约 `2.1 ~ 2.25` 核 |

典型 profile：

```text
[PROFILE] frame=20 fps=7.3 detect=left infer=52.8ms total=167.9ms
[PROFILE] frame=40 fps=7.1 detect=left infer=53.7ms total=175.3ms
[PROFILE] frame=60 fps=7.0 detect=left infer=52.8ms total=180.9ms
```

判断：

```text
完整宽图 + RKNN 路线太慢。
它适合检查检测框和 wide 坐标映射，不适合作为最终实时输出。
瓶颈来自：完整宽图拼接 + RKNN 推理 + 显示 / 坐标映射叠加。
```

推荐程度：

```text
低：作为调试工具保留，不作为最终方案。
```

---

## 8. 当前 camera_movement 主路线测试

测试脚本：

```text
/home/elf/work/basketball/camera_movement/dual_rknn_director_view1920_fast.py
```

运行目的：

```text
验证当前自动运镜主路线性能。
这条路线不生成完整宽幅图，而是直接 remap 输出 1920x1080 view。
```

运行命令：

```bash
DISPLAY=:0 timeout 15s python3 \
  /home/elf/work/basketball/camera_movement/dual_rknn_director_view1920_fast.py \
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
  --print-every 10 \
  --save-dir /tmp/basketball_debug_view1920
```

结果摘要：

| 指标 | 数值 |
|---|---:|
| FPS | `约 20.5 ~ 22.2` |
| 普通帧 total | `约 26 ~ 29 ms` |
| 检测帧 total | `约 148 ~ 155 ms` |
| RKNN infer | `约 57 ~ 65 ms` |
| 1920 view 生成 | `约 26 ~ 29 ms` |
| director 运镜逻辑 | `约 0.2 ~ 0.5 ms` |
| CPU 采样 | 稳定后约 `2.0 ~ 2.3` 核 |

典型 profile：

```text
[PROFILE] frame=20 fps=21.7 detect=left infer=59.8ms director=0.3ms view=27.2ms total=27.5ms
[PROFILE] frame=40 fps=21.4 detect=right infer=60.9ms director=0.3ms view=26.2ms total=26.5ms
[PROFILE] frame=60 fps=20.9 detect=left infer=59.5ms director=0.2ms view=28.4ms total=150.7ms
[PROFILE] frame=100 fps=21.3 detect=right infer=61.5ms director=0.2ms view=28.4ms total=28.7ms
```

CPU 采样示例：

```text
158%
109%
166%
191%
207%
218%
227%
205%
213%
218%
223%
```

判断：

```text
这是当前最推荐的主路线。
它不每帧生成完整宽幅图，而是低频检测 + 高频运镜 + 直接输出 1920x1080 view。
虽然检测帧会出现 150ms 左右尖峰，但总体 FPS 能保持在约 20~22。
```

推荐程度：

```text
高：当前最适合作为最终自动运镜输出路线。
```

---

## 9. “只读取矫正参数做矫正也占 CPU 吗？”的结论

结论：

```text
读取矫正参数 .npz 本身不占 CPU。
真正占 CPU 的是每帧执行 cv2.remap。
```

当前 remap 的工作量：

```text
rectified_size = 2208 x 1242
单路输出像素 ≈ 274 万
双路输出像素 ≈ 548 万 / 帧
```

如果目标 30 FPS：

```text
548 万 x 30 ≈ 1.64 亿像素 / 秒
```

而且每个像素不是简单复制，还要：

```text
查表
从原图取样
双线性插值
写入输出图
```

所以双路 full-frame remap 本身就是大开销。

本次实测：

```text
双路 remap 平均约 32 ms / 帧对
```

30 FPS 单帧预算：

```text
1000 / 30 = 33.3 ms
```

也就是说：

```text
光完整双路 remap 就几乎吃完 30 FPS 的单帧预算。
```

---

## 10. 最终瓶颈判断

### 10.1 USB 采集是否是瓶颈？

结论：

```text
有影响，但不是最终主瓶颈。
```

原因：

```text
同步采集 + 解码约 15 FPS。
但 camera_movement 使用后台线程取最新帧后，最终能到约 20~22 FPS。
说明采集等待可以通过 latest-frame 线程方式缓解。
```

---

### 10.2 MJPG 解码是否是瓶颈？

结论：

```text
是，明确瓶颈之一。
```

依据：

```text
双路 MJPG retrieve/decode 平均约 21~25 ms / 帧对。
```

---

### 10.3 remap 是否是瓶颈？

结论：

```text
是，主要瓶颈之一。
```

依据：

```text
双路 full-frame remap 平均约 32 ms / 帧对。
```

---

### 10.4 拼接融合是否是瓶颈？

结论：

```text
是，完整宽幅拼接的最大常规瓶颈之一。
```

依据：

```text
完整宽图 stitch / alpha 融合平均约 40~42 ms / 帧。
```

---

### 10.5 显示窗口是否是瓶颈？

结论：

```text
会拖慢，尤其不建议显示完整宽幅图。
```

依据：

```text
imshow/waitKey 本身约 2.9 ms。
但 DISPLAY=:0 下完整宽图显示链路整体只有约 3.79 FPS。
```

---

### 10.6 RKNN 是否是瓶颈？

结论：

```text
每帧检测时是瓶颈。
低频检测时可接受。
```

依据：

```text
RKNN 单次推理约 52~65 ms。
完整宽图 + 每帧检测约 7 FPS。
detect_interval=3 + 直接 1920 view 可以达到约 20~22 FPS。
```

---

## 11. 推荐程度汇总

| 路线 | FPS | 推荐程度 | 说明 |
|---|---:|---|---|
| 双路采集 + 解码 | `15 FPS` | 中 | 可作为基础基线，不是最终业务链路 |
| 双路采集 + remap | `15 FPS` | 中 | remap 开始成为明显瓶颈 |
| 完整宽幅拼接 | `10 FPS` | 低 | 适合调试，不适合最终实时输出 |
| 完整宽幅拼接 + LCD 显示 | `3.8 FPS` | 很低 | 只适合临时观察 |
| 完整宽幅拼接 + RKNN | `7 FPS` | 低 | 检测坐标调试可用，不适合最终输出 |
| 当前 `dual_rknn_director_view1920_fast.py` | `20~22 FPS` | 高 | 当前最推荐主路线 |

---

## 12. 最终结论

当前完整宽幅链路的主要耗时可以概括为：

```text
MJPG 解码 约 21~25 ms
    +
双路 remap 约 32 ms
    +
完整宽幅拼接融合 约 40~42 ms
    =
完整拼接约 97~100 ms / 帧
    =
约 10 FPS
```

如果再加 RKNN 每帧检测：

```text
完整拼接约 100 ms
    +
RKNN 约 53~60 ms
    +
显示 / 坐标映射等
    =
约 170~180 ms / 帧
    =
约 7 FPS
```

因此，最终实时路线不应走：

```text
每帧完整双路 remap
每帧生成完整宽幅图
每帧 RKNN 检测
每帧显示完整宽幅图
```

当前最合理的路线是：

```text
后台双路采集
    ↓
低频 RKNN 检测
    ↓
导播算法每帧更新
    ↓
只 remap 最终需要的 1920x1080 view
    ↓
输出 / 显示最终视口
```

也就是继续以：

```text
camera_movement/dual_rknn_director_view1920_fast.py
```

作为主线。

---

## 13. 后续优化方向记录

本次没有做优化，只记录后续可能方向：

1. 避免每帧完整宽图拼接。
2. 避免每帧 full-frame 双路 remap。
3. 优先保留 ROI / view 级 remap。
4. RKNN 保持低频检测，例如 `--detect-interval 3` 或 `4`。
5. LCD 显示只显示最终 1920 view，不显示完整宽幅图。
6. 调试时才使用完整宽图显示和完整宽图 RKNN 坐标验证。
7. 若后续要继续提速，可考虑 RGA / C++ / 硬件图像处理，但这不属于本次验证范围。

---

## 14. 本次日志一句话总结

```text
当前卡点主要不是“读取参数”，而是每帧 full-frame remap、完整宽幅拼接融合、MJPG 解码和每帧 RKNN 叠加；最终推荐走低频检测 + 直接 1920 view remap 的 camera_movement fast 路线。
```
