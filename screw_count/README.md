# 螺丝计数项目说明（screw_count）

本目录用于完成 Lab2 的螺丝计数任务。默认方案基于 YOLO 检测，将每张图中 5 类螺丝分别计数，最终输出评分脚本要求的 `result.npy` 与 `time.txt`。

## 1. 任务与输出格式

程序入口是 `run.py`：

```bash
python run.py --data_dir /path/to/test_images --output_path ./result.npy --output_time_path ./time.txt
```

输出：

- `result.npy`：`numpy.load(..., allow_pickle=True).item()` 
- 字典 key：图片文件名（不含后缀）。
- 字典 value：长度为 5 的列表，顺序固定为 `[Type_1, Type_2, Type_3, Type_4, Type_5]`。
- `time.txt`：处理整个文件夹总耗时（秒）。

## 2. 环境要求

- Python >= 3.9

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 目录结构

```text
screw_count/
├── run.py                    # 评分入口：批量预测并导出 result.npy + time.txt
├── augment_dataset.py        # 离线数据增强 + train/val 切分
├── train_yolo.py             # YOLO 训练脚本（自动生成 screws.yaml）
├── predict_visualize.py      # 推理并保存可视化检测结果
├── screws.yaml               # 训练数据描述（可由训练脚本自动重写）
├── requirements.txt
├── images/                   # 原始训练图像
├── labels/                   # YOLO 标注（与 images 同名 txt）
├── augmented/                # 增强后数据与切分结果
├── runs/                     # 训练输出
├── vis/                      # 可视化输出
└── weights/
    ├── best.pt               # 推理默认使用的权重
    ├── yolo11n.pt            # 预训练权重（可选）
    └── yolo11s.pt            # 预训练权重（可选）
```

## 4. 快速开始

### 4.1 直接推理

```bash
python run.py \
  --data_dir /path/to/test_images \
  --output_path ./result.npy \
  --output_time_path ./time.txt
```

可选参数：

- `--weights`：默认 `weights/best.pt`
- `--conf`：默认 `0.25`
- `--iou`：默认 `0.45`
- `--imgz`: 默认`1280`

### 4.2 可视化推理结果（可选）

```bash
python predict_visualize.py --data_dir ./data/test --weights ./weights/best.pt --output_dir ./vis
```

脚本会在 `vis/` 下保存带框结果图，便于排查漏检/误检。

## 5. 训练复现流程

### 步骤 1：离线数据增强并切分

```bash
python augment_dataset.py \
  --img_dir ./images \
  --label_dir ./labels \
  --output_dir ./augmented \
  --aug_per_image 50 \
  --val_ratio 0.15 \
  --seed 42
```

说明：

- 先复制原图，再生成增强图。
- 按“原图分组”切分 train/val，避免同一原图及其增强图分到不同集合造成数据泄漏。

### 步骤 2：训练 YOLO

```bash
python train_yolo.py \
  --data_dir ./augmented \
  --model yolo11n \
  --epochs 150 \
  --batch 8 \
  --freeze 10 \
  --output_dir ./runs
```

训练输出权重默认路径：

```text
./runs/yolo11n_screws/weights/best.pt
```

用于提交推理时，请将最优权重复制为：

```text
./weights/best.pt
```
