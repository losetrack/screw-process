# 螺丝实例分割（screw_segment）

YOLO 检测 + 分割后端的混合实例分割流程，用于 Lab3 螺丝分割任务。

## 整体架构

```
输入图像 → ScrewDetector (YOLO) → 检测框 → Segmentor → 实例掩膜 → Visualizer → 可视化输出
```

三个核心组件通过统一接口协作：

- **`ScrewDetector`** (`detector.py`): YOLO 检测，返回每个螺丝的边界框、类别和置信度
- **`ThresholdSegmentor`** / **`SamSegmentor`**: 在检测框内执行分割，输出 `instance_map`（uint16 标签图）
- **`SegmentationVisualizer`** (`visualizer.py`): 将实例掩膜渲染为彩色可视化图
- **`SegmentationPipeline`** (`segment.py`): 编排完整的检测 → 分割 → 可视化流程

## 快速开始

### SAM 分割

```bash
python screw_segment/segment.py \
  --data_dir ./data \
  --output_dir ./output \
  --segmentor sam \
  --detector_weights ./weights/best.pt \
  --sam_weights ./weights/sam_vit_l_0b3195.pth
```

### 传统 CV 分割

```bash
python screw_segment/segment.py \
  --data_dir ./data \
  --output_dir ./output \
  --segmentor threshold \
  --seg_method otsu   # otsu | adaptive | watershed
```

## 分割后端

### 1. 传统 CV 阈值分割 (`threshold_segmentor.py`)

在 YOLO 检测框 ROI 内进行分割。三种方法：

| 方法 | 适用场景 | 原理 |
|------|----------|------|
| `otsu` | 标准光照 | Otsu 全局阈值 + 形态学清理 |
| `adaptive` | 光照不均 | 自适应高斯阈值 |
| `watershed` | 螺丝粘连 | 距离变换 + 分水岭算法分离粘连目标 |

### 2. SAM 分割 (`sam_segmentor.py`)

Meta SAM 模型，以 YOLO 检测框作为提示进行分割。

- 自动从权重文件名推断模型变体（`vit_h` / `vit_l` / `vit_b`）
- 支持 `multimask_output` 多候选并选择得分最高的 mask
- 检测框可扩展像素数（`box_margin`）以提供更多上下文

SAM 权重需下载至 `screw_segment/weights/`，例如 `sam_vit_l_0b3195.pth`。

## 输出说明

每张输入图像 `{stem}.png` 生成以下文件：

| 文件 | 说明 |
|------|------|
| `{stem}_mask.png` | 彩色掩膜叠加图（原图 + 半透明掩膜） |
| `{stem}_instances.png` | 伪彩色实例标签图（可选，`--save_instance_map`） |
| `{stem}_instances.npy` | uint16 实例标签数组（可选，`--save_instance_map`） |

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 输入图像目录 |
| `--output_dir` | `./output` | 输出目录 |
| `--segmentor` | `sam` | 分割算法选择：`threshold` 或 `sam` |
| `--detector_weights` | `./weights/best.pt` | YOLO 权重路径 |
| `--sam_weights` | `./weights/sam_vit_l_0b3195.pth` | SAM 权重路径 |
| `--conf` | `0.25` | 检测置信度阈值 |
| `--iou` | `0.45` | NMS IoU 阈值 |
| `--imgsz` | `640` | YOLO 输入尺寸 |
| `--seg_method` | `otsu` | 分割方法（仅 threshold 模式）：`otsu` / `adaptive` / `watershed` |
| `--margin` | `5` | 检测框扩展像素数 |
| `--save_instance_map` | `False` | 是否保存实例标签图和 npy 数据 |

## 依赖

```bash
pip install ultralytics opencv-python numpy torch torchvision
# SAM 额外需要:
pip install segment-anything
```

## 目录结构

```
screw_segment/
├── segment.py               # 主入口，编排检测+分割流程
├── detector.py              # YOLO 检测封装
├── threshold_segmentor.py   # 传统 CV 分割（Otsu/自适应/分水岭）
├── sam_segmentor.py         # SAM 分割（检测框提示）
├── visualizer.py            # 结果可视化
├── weights/                 # YOLO 和 SAM 权重目录
├── data/                    # 输入图像（已被 gitignore）
└── output/                  # 输出结果（已被 gitignore）
```

## Lab3 评分标准

- 50% 结果（−2 每遗漏螺丝，−1 每误检/不完整）
- 50% 报告
