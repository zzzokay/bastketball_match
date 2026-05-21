# USB Camera Chessboard Calibration

USB 摄像头棋盘格标定 + 实时畸变矫正工具，适用于 Linux / RK3588 等嵌入式平台。

## 环境要求

- Python 3
- OpenCV（需支持 V4L2，无需 GStreamer）
- NumPy

```bash
pip install opencv-python numpy
```

## 标定板参数

| 参数 | 值 |
|------|-----|
| 方格阵列 | 12 x 9 |
| 内角点数 | 11 x 8 |
| 方格边长 | 3 mm |

## 使用流程

### 1. 采集标定图片

打开摄像头实时预览，手动拍摄棋盘格图片：

```bash
python3 calib_usb_camera.py --mode capture
```

- 按 `c` — 保存当前检测到角点的标定图
- 按 `s` — 直接保存原图（不检测角点）
- 按 `q` / `Esc` — 退出

建议采集 **15~30 张**不同角度、位置的图片。图片保存在 `calib_images/` 目录下。

### 2. 计算标定参数

```bash
python3 calib_usb_camera.py --mode calibrate
```

程序会读取 `calib_images/` 中的标定图，计算相机内参和畸变系数，结果保存到 `camera_calib.npz`。

误差参考：
- < 0.3 像素 — 很好
- 0.3 ~ 0.8 像素 — 正常可用
- 0.8 ~ 1.5 像素 — 勉强可用
- \> 1.5 像素 — 建议重新采集

### 3. 实时畸变矫正

```bash
python3 calib_usb_camera.py --mode undistort
```

- 按 `o` — 切换原图/矫正图上下对比显示
- 按 `s` — 保存当前矫正图像
- 按 `q` / `Esc` — 退出

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `/dev/video41` | 摄像头设备节点 |
| `--width` | `1920` | 采集宽度 |
| `--height` | `1080` | 采集高度 |
| `--fps` | `30` | 帧率 |
| `--no-mjpg` | `false` | 禁用 MJPG，使用默认格式 |
| `--dist-scale` | `1.0` | 畸变矫正强度（0~1） |
| `--alpha` | `0.0` | 0=裁黑边，1=保留全部视野 |
| `--crop` | `false` | 矫正后裁掉黑边 |
| `--display-scale` | `0.5` | 显示窗口缩放比例 |
| `--detect-every` | `5` | 采集预览每隔 N 帧检测角点 |
| `--detect-scale` | `0.5` | 采集预览角点检测缩放比例 |

## 项目文件

```
basketball/
├── calib_usb_camera.py          # 主程序
├── calib_images/                # 标定图片目录
├── camera_calib.npz             # 标定结果（默认）
├── camera_calib_fixed_center.npz # 标定结果（固定主点）
└── undistorted_*.jpg            # 矫正示例输出
```
