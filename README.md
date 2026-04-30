# ScrewProcess — 工业螺丝视觉检测与计数

基于计算机视觉的螺丝检测、计数与分割项目，涵盖图像校正、目标检测、实例分割和视频多目标跟踪。

## 模块概览

| 模块 | 任务 | 技术栈 | 文档 |
|------|------|--------|------|
| homograpy_restore | 单应性恢复与多余螺丝去除 | SIFT / ORB / RANSAC | [README](homograpy_restore/README.md) |
| screw_count | YOLO 螺丝检测与计数 | Ultralytics YOLO11 | [README](screw_count/README.md) |
| screw_segment | 螺丝实例分割 | YOLO + SAM / 传统 CV | [README](screw_segment/README.md) |
| screw_video | 视频螺丝跟踪与计数 | YOLO + ByteTrack + OSNet Re-ID | [README](screw_video/README.md) |

## 环境要求

- Python >= 3.9
- 推荐 Conda 环境：
```bash
  conda create -n screw_process python=3.10
  conda activate screw_process
```
- 模块间依赖独立，按需安装相应依赖 `pip install -r requirements.txt`

### 通用依赖

```bash
pip install opencv-python numpy torch torchvision
```

### 各模块额外依赖

- **screw_count / screw_segment / screw_video**: `pip install ultralytics`
- **screw_segment (SAM)**: 下载 SAM 权重至 `screw_segment/weights/`
- **screw_video**: `pip install boxmot torchreid`

## 模块详情

### 1. 单应性恢复 (`homograpy_restore/`)

将透视变形图像校正为模板视角，可选去除多余螺丝。

```bash
python homograpy_restore/restore.py \
  --template data/template.png \
  --input_dir data/ \
  --output_dir restored_images/
```

### 2. 螺丝检测计数 (`screw_count/`)

YOLO 检测 5 类螺丝并计数，输出评分所需格式。

```bash
python screw_count/run.py \
  --data_dir /path/to/test_images \
  --output_path ./result.npy \
  --output_time_path ./time.txt
```

### 3. 实例分割 (`screw_segment/`)

YOLO 检测 + 分割后端（阈值分割 / SAM），生成实例掩膜。

```bash
python screw_segment/segment.py \
  --data_dir ./data --output_dir ./output \
  --segmentor [threshold|sam]
```

### 4. 视频跟踪计数 (`screw_video/`)

YOLO 检测 + ByteTrack 跟踪 + OSNet Re-ID 跨轨迹匹配，实现视频螺丝去重计数。

```bash
python screw_video/run.py \
  --data_dir ./vedio_exp \
  --output_path ./result.npy \
  --output_time_path ./time.txt \
  --mask_output_path ./mask_folder
```

## 输出格式

各模块输出格式一致：

- **result.npy**: `numpy.load(..., allow_pickle=True).item()` 得到字典，key 为文件名（不含后缀），value 为 `[Type_1, Type_2, Type_3, Type_4, Type_5]` 计数列表
- **time.txt**: 处理总耗时（秒）

## 项目结构

```
ScrewProcess/
├── homograpy_restore/     # Lab1: 单应性恢复
│   ├── restore.py
│   └── README.md
├── screw_count/           # Lab2: 检测计数
│   ├── run.py
│   ├── train_yolo.py
│   ├── augment_dataset.py
│   └── README.md
├── screw_segment/         # Lab3: 实例分割
│   ├── segment.py
│   ├── detector.py
│   ├── threshold_segmentor.py
│   ├── sam_segmentor.py
│   └── visualizer.py
├── screw_video/           # 视频跟踪计数
│   ├── run.py
│   ├── test.py
│   ├── detector.py
│   ├── tracker.py
│   ├── reid.py
│   ├── counter.py
│   └── visualizer.py
├── CLAUDE.md
└── README.md
```

## 开发说明

- 各模块相互独立，可单独运行
- 所有模块均使用 OpenCV BGR 格式处理图像
- CUDA 自动检测，默认优先使用 GPU
- 详细指南见 `CLAUDE.md`
