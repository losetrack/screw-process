# 螺丝计数 Lab2 - 环境配置与运行说明

## 环境要求

- Python 3.9+
- CUDA 11.8（推荐，CPU也可运行但较慢）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 文件结构

```
code/
├── run.py                # 推理入口（自动评分用）
├── train_yolo.py         # 训练脚本（复现用）
├── augment_dataset.py    # 数据增强脚本（复现用）
├── weights/
│   └── best.pt           # 训练好的 YOLOv8 权重
├── requirements.txt
└── README.md
```

## 一键推理（评分用）

```bash
python run.py \
  --data_dir /path/to/test_images \
  --output_path ./result.npy \
  --output_time_path ./time.txt
```

输出：
- `result.npy`：字典，key=图片名(无后缀)，value=[Type1数, Type2数, ..., Type5数]
- `time.txt`：总推理时间（秒）

## 复现训练（可选）

### 1. 数据增强

```bash
python augment_dataset.py \
  --img_dir ./images \
  --label_dir ./labels \
  --output_dir ./augmented \
  --aug_per_image 50
```

### 2. 训练

```bash
python train_yolo.py \
  --data_dir ./augmented \
  --model yolov8s \
  --epochs 150 \
  --freeze 10
```

训练完成后将 `runs/yolov8s_screws/weights/best.pt` 复制到 `weights/best.pt`。

## 调参建议

| 参数 | 小数据推荐值 | 说明 |
|------|-------------|------|
| `--conf` | 0.25 | 置信度阈值，偏低可减少漏检 |
| `--iou`  | 0.45 | NMS阈值，螺丝密集时可调低至0.3 |
| `--freeze` | 10 | 冻结骨干层数，数据少时用10 |
