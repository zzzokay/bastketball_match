# RK3588 + RKNN + YOLO 模型部署排查 README

本文档记录本次在 **RK3588 / ELF2** 上部署 YOLO 模型时遇到的问题、定位过程、解决办法和后续注意事项。

当前已验证：使用旧项目的 `best_2.rknn` 可以在 `/dev/video43` USB 摄像头上稳定识别人，并能正常退出程序。

---

## 1. 当前稳定运行命令

目前已经验证可用的是：

```bash
cd ~/work/basketball

python3 infer_person.py \
  --device /dev/video43 \
  --model /home/elf/work/basketball/model/best_2.rknn \
  --conf 0.25 \
  --nms 0.45
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--device /dev/video43` | 使用 USB 摄像头设备节点 |
| `--model best_2.rknn` | 使用已验证可运行的旧模型 |
| `--conf 0.25` | 置信度阈值，低于该值的框会被过滤 |
| `--nms 0.45` | NMS IoU 阈值，用于去除重复框 |

退出方式：

```text
q / ESC：窗口有焦点时正常退出
Ctrl+C：终端中断退出
如果卡死：另开终端执行 pkill -9 -f infer_person.py
```

---

## 2. 当前环境信息

板端日志显示：

```text
rknn-toolkit-lite2 version: 2.3.2
librknnrt version: 2.1.0
RKNN Driver version: 0.9.8
target platform: rk3588
```

注意：虽然 Python 包 `rknn-toolkit-lite2` 是 2.3.2，但底层实际运行库 `librknnrt` 是 2.1.0。模型转换时应尽量与板端 runtime 保持一致。

已验证可运行的旧模型：

```text
best_2.rknn
toolkit version: 2.1.0+708089d1
compiler version: 2.1.0
output shape: (1, 5, 8400)
```

存在问题的新模型：

```text
basketball_player_2.1.0.rknn
toolkit version: 2.1.0+708089d1
compiler version: 2.1.0
output shape: (1, 84, 8400)
cls_scores min/max/mean: 0.0 0.0 0.0
```

---

## 3. 模型输出 shape 的含义

YOLO 检测模型常见输出格式：

```text
(1, C, 8400)
```

其中：

```text
C = 4 + 类别数
```

因此：

| 输出 shape | 含义 |
|---|---|
| `(1, 5, 8400)` | 单类别模型，`5 = 4 box + 1 class` |
| `(1, 84, 8400)` | COCO 80 类模型，`84 = 4 box + 80 class` |

本次确认：

```text
best_2.rknn -> (1, 5, 8400) -> 单类模型
basketball_player.rknn / basketball_player_2.1.0.rknn -> (1, 84, 8400) -> COCO 80 类模型
```

---

## 4. 类别列表确认

通过 `basketball_player.pt` 查询到类别数量为 80：

```text
0 person
1 bicycle
2 car
3 motorcycle
...
32 sports ball
...
79 toothbrush
```

所以：

```text
如果只识别人：PERSON_CLASS_ID = 0
如果识别篮球：sports ball = 32
```

当前代码中：

```python
PERSON_CLASS_ID = 0
```

是正确的，因为 `0 = person`。

---

## 5. 问题一：一开始出现很多乱框

### 现象

模型运行后画面出现大量错误框，看起来不像在正常识别。

### 原因

YOLO11 输出为：

```text
(1, 84, 8400)
```

原来直接使用：

```python
output_2d = output.reshape(-1, 84)
```

这是错误的。`reshape()` 只是按内存硬拆数据，不会按 YOLO 输出语义重新排列通道，导致 box 坐标和类别分数错位。

### 正确做法

应先去掉 batch 维度，再转置：

```python
output = outputs[0]
pred = np.squeeze(output)      # (84, 8400)
output_2d = pred.T             # (8400, 84)
```

为了同时兼容 `(1, 5, 8400)` 和 `(1, 84, 8400)`，推荐使用通用写法：

```python
pred = np.squeeze(output)

if pred.ndim != 2:
    print(f"[警告] 暂不支持的输出维度: {output.shape}")
    return []

# NCHW/CHW: (C, N) -> 转置成 (N, C)
# NHWC/HWC: (N, C) -> 直接使用
if pred.shape[0] < pred.shape[1] and pred.shape[0] >= 5:
    output_2d = pred.T
elif pred.shape[1] >= 5:
    output_2d = pred
else:
    print(f"[警告] 无法解析的 YOLO 输出 shape: {output.shape}")
    return []
```

---

## 6. 问题二：修正输出解析后没有框

### 现象

修正输出解析后乱框消失，但完全识别不到人。

调试输出：

```text
cls_scores min/max/mean: 0.0 0.0 0.0
person score > 0.25: 0
```

### 原因

这说明模型输出的类别分数本身就是全 0。这种情况下不是阈值问题，也不是 NMS 问题，后处理无法救。

### 判断方法

如果看到：

```text
cls_scores max = 0.0
```

说明模型输出异常。此时不要继续盲目调：

```bash
--conf 0.25
--conf 0.05
--conf 0.01
```

而应该排查：

```text
1. RKNN 模型转换是否异常
2. INT8 量化是否异常
3. ONNX 导出是否正确
4. RKNN Toolkit 与板端 runtime 是否兼容
```

---

## 7. 问题三：RKNN 模型版本不匹配

### 现象

一开始新模型日志出现：

```text
RKNN Model version: 2.3.2 not match with rknn runtime version: 2.1.0
```

### 原因

模型由 RKNN-Toolkit2 2.3.2 转换，但板端底层 `librknnrt` 是 2.1.0。

### 解决办法

使用 RKNN-Toolkit2 2.1.0 重新转换：

```bash
python -m pip list | grep rknn
```

确认虚拟机中为：

```text
rknn-toolkit2 2.1.0+708089d1
```

然后重新转换：

```bash
cd rknn_model_zoo-main/examples/yolo11/python

python convert.py \
  ../model/basketball_player.onnx \
  rk3588 \
  i8 \
  ../model/basketball_player_2.1.0.rknn
```

板端再运行：

```bash
python3 infer_person.py \
  --device /dev/video43 \
  --model /home/elf/work/basketball/model/basketball_player_2.1.0.rknn \
  --conf 0.25 \
  --nms 0.45
```

### 注意

重新转换后版本匹配了，但 `basketball_player_2.1.0.rknn` 仍然出现 `cls_scores 全 0`。所以版本不匹配只是其中一个问题，不是最终原因。

---

## 8. 问题四：转换日志中有危险错误

转换新模型时出现过：

```text
W build: found outlier value, this may affect quantization accuracy
E RKNN: REGTASK: The bit width of field value exceeds the limit
```

### 含义

`found outlier value` 表示模型权重存在离群值，INT8 量化可能损失严重。

`REGTASK bit width exceeds the limit` 表示 RKNN 编译阶段出现寄存器任务字段超限。即使最后打印：

```text
Export rknn model done
```

也不能说明模型一定可用。

### 结论

`basketball_player_2.1.0.rknn` 虽然能加载、能推理，但输出类别分数全 0，因此不能作为有效模型继续调代码。

---

## 9. 为什么 `best_2.rknn` 能跑

使用 `best_2.rknn` 时输出：

```text
output shape: (1, 5, 8400)
cls_scores min/max/mean: 0.0 0.84228515625 0.0008259546
person score > 0.25: 8
```

说明：

```text
摄像头正常
OpenCV 采集正常
letterbox 预处理正常
RGB/BGR 当前设置可用
RKNN Runtime 正常
NMS 正常
绘制检测框正常
退出逻辑正常
```

所以当前代码和板端环境主流程是通的。问题主要集中在新 YOLO11 模型转换链路。

---

## 10. 阈值和 NMS 的经验值

当前推荐：

```bash
--conf 0.25 --nms 0.45
```

调参建议：

| 现象 | 调整 |
|---|---|
| 框太多 | 提高 `--conf`，如 0.35 / 0.45 |
| 漏检 | 降低 `--conf`，如 0.2 / 0.15 |
| 重叠框太多 | 降低 `--nms`，如 0.35 |
| 同一目标被多个框重复标出 | 降低 `--nms` 或限制 max_det |
| 完全没有框且 `cls_scores max = 0` | 不要调阈值，排查模型转换 |

不要一开始就用：

```bash
--conf 0.01
```

否则容易显示大量低质量候选框。

---

## 11. sigmoid 的注意事项

YOLOv8 / YOLO11 的输出有时已经是 0~1 概率，有时是 raw logits。

推荐只在分数超出 `[0, 1]` 时做 sigmoid：

```python
if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
    cls_scores = sigmoid(cls_scores)
```

不要无条件做：

```python
cls_scores = sigmoid(cls_scores)
```

否则如果原始分数是：

```text
0.003
```

经过 sigmoid 会变成：

```text
0.5007
```

会导致大量低分框变成 0.5 左右，出现满屏框。

---

## 12. 预处理流程

当前稳定流程：

```text
摄像头 BGR 图像
↓
letterbox 等比缩放 + 灰色填充到 640x640
↓
BGR 转 RGB
↓
添加 batch 维度，变成 NHWC
↓
输入 RKNN
```

核心代码：

```python
canvas, ratio, pad_w, pad_h = letterbox(image_bgr, MODEL_INPUT_SIZE)

if self.use_rgb:
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

blob = np.expand_dims(canvas, axis=0)
blob = np.ascontiguousarray(blob)

outputs = self.rknn.inference(inputs=[blob], data_format=['nhwc'])
```

如果怀疑颜色通道不对，可以测试：

```bash
python3 infer_person.py ... --bgr-input
```

但当前 `best_2.rknn` 默认 RGB 路线已经验证正常。

---

## 13. Ctrl+C 退出问题

### 现象

程序运行时按 `Ctrl+C`，会反复打印：

```text
Catch exception when running RKNN model
KeyboardInterrupt
```

甚至无法退出。

### 原因

`RKNNLite.inference()` 内部正在执行 NPU 推理时，`Ctrl+C` 可能先被 RKNNLite 捕获并打印异常，导致退出不干净。

### 解决办法

在 `detect()` 中包裹推理：

```python
try:
    outputs = self.rknn.inference(inputs=[blob], data_format=['nhwc'])
except KeyboardInterrupt:
    raise
except Exception as e:
    print(f"[错误] RKNN 推理失败: {e}")
    return []
```

在主函数中确保释放资源：

```python
try:
    while True:
        ...
except KeyboardInterrupt:
    print("\n[信息] Ctrl+C 退出")
finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[信息] 程序已退出")
```

如果仍然卡死，另开终端：

```bash
pkill -9 -f infer_person.py
```

如果摄像头被占用：

```bash
fuser -v /dev/video43
sudo fuser -k /dev/video43
```

---

## 14. Windows PowerShell 注意事项

### 错误写法

下面是 Linux/bash 写法，PowerShell 不支持：

```bash
python3 - <<'PY'
```

在 PowerShell 中直接输入 Python 代码也不行：

```python
from ultralytics import YOLO
```

PowerShell 会把它当作 PowerShell 命令。

### 正确写法一：进入 Python 解释器

```powershell
conda activate yolo
python
```

看到：

```text
>>>
```

后再输入：

```python
from ultralytics import YOLO

model = YOLO("D:/deeplearning/model/basketball_player.pt")

print("类别数量:", len(model.names))
print("类别列表:")
for k, v in model.names.items():
    print(k, v)
```

### 正确写法二：PowerShell 一行命令

```powershell
python -c "from ultralytics import YOLO; model=YOLO('D:/deeplearning/model/basketball_player.pt'); print(model.names)"
```

### Conda 激活环境

错误：

```powershell
conda yolo
```

正确：

```powershell
conda activate yolo
```

---

## 15. 新 YOLO11 模型后续修复路线

如果还要继续修 `basketball_player_2.1.0.rknn`，建议按以下步骤。

### 第一步：先转 FP 模型，不要直接 i8

```bash
python convert.py \
  ../model/basketball_player.onnx \
  rk3588 \
  fp \
  ../model/basketball_player_fp_2.1.0.rknn
```

板端测试：

```bash
python3 infer_person.py \
  --device /dev/video43 \
  --model /home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn \
  --conf 0.25 \
  --nms 0.45
```

判断：

```text
FP 正常，i8 不正常：
    说明量化校准数据不合适。

FP 也不正常：
    说明 ONNX 导出或 YOLO11 转 RKNN 兼容性有问题。
```

### 第二步：如果 FP 正常，再重新做 i8 量化

采集真实篮球场图像作为校准集，建议 100～300 张，包含：

```text
篮球场远景
球员近景
多人场景
遮挡场景
不同光照
不同角度
运动模糊
空场景
```

生成校准列表：

```bash
find calib_images -name "*.jpg" > basketball_calib.txt
```

在 `convert.py` 中将默认 dataset 改为：

```python
DATASET_PATH = "basketball_calib.txt"
```

然后重新转换 i8。

---

## 16. 标准排查流程

以后遇到 RKNN YOLO 模型异常，按这个顺序查：

```text
1. 看 output shape
2. 看 cls_scores min/max/mean
3. 看类别 ID 是否正确
4. 看 box 坐标范围是否正常
5. 调 conf 和 nms
6. 检查 RGB / BGR
7. 检查 RKNN Toolkit 与 runtime 版本
8. 检查转换日志是否有 REGTASK / outlier
9. 先测 FP，再测 i8
10. 使用真实场景校准数据重新量化
```

关键判断：

```text
如果 cls_scores max = 0：
    模型输出异常，后处理无法解决。

如果 cls_scores max 正常但框太多：
    调 conf / nms，检查 sigmoid 是否重复。

如果 cls_scores 正常但框位置乱：
    检查 reshape / transpose、letterbox 坐标还原。

如果旧模型能跑，新模型不能跑：
    优先排查新模型导出和转换。
```

---

## 17. 当前建议

短期项目开发：

```text
继续使用 best_2.rknn
命令参数固定为 --conf 0.25 --nms 0.45
先完成摄像头、显示、报警、ROI、业务逻辑等功能
```

新 YOLO11 模型：

```text
单独排查
先转 FP
FP 正常后再做 i8
i8 量化必须使用真实篮球场校准图片
看到 REGTASK 错误时不要盲目信任导出的模型
```

---

## 18. 最终稳定命令备份

```bash
cd ~/work/basketball

python3 infer_person.py \
  --device /dev/video43 \
  --model /home/elf/work/basketball/model/best_2.rknn \
  --conf 0.25 \
  --nms 0.45
```

如果进程卡死：

```bash
pkill -9 -f infer_person.py
```

如果摄像头被占用：

```bash
fuser -v /dev/video43
sudo fuser -k /dev/video43
```

---

## 19. 本次经验总结

这次问题的关键经验：

```text
乱框不一定是 NMS 问题，可能是输出 reshape/transpose 错。
没框不一定是阈值问题，要先看 cls_scores 是否全 0。
能导出 rknn 不代表模型可用。
i8 量化前应该先验证 FP 模型。
旧模型能跑是非常好的对照组。
PowerShell 和 Linux 终端命令不能混用。
RKNNLite 推理程序必须处理退出和资源释放。
```

当前已经跑通的链路：

```text
USB 摄像头 /dev/video43
OpenCV 采集
letterbox 预处理
RKNNLite 推理
YOLO 后处理
NMS
画框显示
Ctrl+C 正常退出
```

后续可以在此基础上继续扩展 ROI、报警、篮球场区域检测和多摄像头拼接等功能。
