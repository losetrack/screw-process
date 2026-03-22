"""
YOLOv8 训练脚本 - 螺丝计数任务
用法:
    python train_yolo.py --data_dir ./augmented --epochs 150 --model yolo11s

依赖:
    pip install ultralytics opencv-python albumentations
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import yaml


# ──────────────────────── 生成 dataset YAML ──────────────────────────

def create_dataset_yaml(data_dir, output_path='screws.yaml'):
    data_dir = Path(data_dir).resolve()

    config = {
        'path': str(data_dir),
        'train': 'train/images',
        'val':   'val/images',

        # 5种螺丝类别（按作业要求顺序）
        'nc': 5,
        'names': ['Type_1', 'Type_2', 'Type_3', 'Type_4', 'Type_5'],
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"Dataset YAML saved: {output_path}")
    print(f"  path : {data_dir}")
    print(f"  train: {data_dir / 'train/images'}")
    print(f"  val  : {data_dir / 'val/images'}")
    return output_path


# ──────────────────────── 训练配置 ───────────────────────────────────

def get_train_args(model_name, data_yaml, epochs, batch, imgsz, freeze, output_dir):
    """
    返回传给 model.train() 的参数字典。
    针对小数据集 + 俯视螺丝场景做了专项调整。
    """
    return dict(
        data      = data_yaml,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        device    = 'cuda' if _cuda_available() else 'cpu',

        # ── 迁移学习：冻结骨干网络前 N 层 ──────────────────────────
        # 小数据集建议 freeze=10（冻结 backbone 大部分），只训练 neck+head
        # 数据多了之后可改为 freeze=0 全量微调
        freeze    = freeze,

        # ── 优化器 ──────────────────────────────────────────────────
        optimizer = 'AdamW',
        lr0       = 0.001,          # 初始学习率
        lrf       = 0.01,           # 最终学习率 = lr0 * lrf
        warmup_epochs = 5,
        weight_decay  = 0.0005,
        momentum      = 0.937,

        # ── 内置数据增强（在 augment_dataset.py 之外的第二层增强）──
        # 这些是 YOLO 训练时 on-the-fly 的增强，与离线增强互补
        augment   = True,
        degrees   = 180.0,          # 旋转（俯视图全角度有效）
        fliplr    = 0.5,
        flipud    = 0.5,
        mosaic    = 0.5,            # 4张拼1张，提升小目标识别
        mixup     = 0.1,
        hsv_h     = 0.02,
        hsv_s     = 0.5,
        hsv_v     = 0.4,
        scale     = 0.5,
        translate = 0.1,
        perspective = 0.001,
        copy_paste  = 0.1,          # 跨图粘贴，对小数据集有效

        # ── 训练策略 ────────────────────────────────────────────────
        patience     = 30,          # early stopping 等待轮数
        save         = True,
        save_period  = 10,          # 每10个epoch保存一次
        val          = True,
        plots        = True,        # 保存混淆矩阵、PR曲线等

        # ── 损失权重（螺丝检测以分类准确为主，适当加大 cls）────────
        cls   = 1.5,                # 分类损失权重（默认0.5，加大区分5类）
        box   = 7.5,
        dfl   = 1.5,

        # ── 输出路径 ─────────────────────────────────────────────────
        project = output_dir,
        name    = f'{model_name}_screws',
        exist_ok = True,

        # ── 其他 ─────────────────────────────────────────────────────
        workers   = 4,
        seed      = 42,
        verbose   = True,
        amp       = True,           # 混合精度加速（2080 Ti 支持）
    )


# ──────────────────────── 验证 & 导出 ────────────────────────────────

def validate_and_export(model, output_dir, model_name, data_yaml):
    print("\n" + "="*50)
    print("验证最优模型 (best.pt)...")
    best_pt = Path(output_dir) / f'{model_name}_screws' / 'weights' / 'best.pt'

    if not best_pt.exists():
        print(f"  [warn] best.pt not found at {best_pt}")
        return

    from ultralytics import YOLO
    best_model = YOLO(str(best_pt))

    # 在验证集上输出详细指标
    results = best_model.val(data=data_yaml, imgsz=640, verbose=True)

    print("\n各类别 AP50:")
    names = ['Type_1', 'Type_2', 'Type_3', 'Type_4', 'Type_5']
    if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
        for idx, cls_idx in enumerate(results.box.ap_class_index):
            ap = results.box.ap50[idx]
            print(f"  {names[cls_idx]:8s}: AP50 = {ap:.4f}")

    print(f"\nmAP50   : {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")


# ──────────────────────── 工具函数 ───────────────────────────────────

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def print_dataset_stats(data_dir):
    data_dir = Path(data_dir)
    for split in ['train', 'val']:
        img_dir = data_dir / split / 'images'
        lbl_dir = data_dir / split / 'labels'
        if not img_dir.exists():
            continue
        imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        counts = [0] * 5
        for lbl in lbl_dir.glob('*.txt'):
            with open(lbl) as f:
                for line in f:
                    cls = int(line.strip().split()[0])
                    if 0 <= cls < 5:
                        counts[cls] += 1
        print(f"  {split:5s}: {len(imgs):4d} images | "
              f"Type_1={counts[0]} Type_2={counts[1]} Type_3={counts[2]} "
              f"Type_4={counts[3]} Type_5={counts[4]}")


# ──────────────────────── 主流程 ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 螺丝检测训练脚本')
    parser.add_argument('--data_dir',   default='./augmented',
                        help='augment_dataset.py 的输出目录（包含 train/val 子目录）')
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--batch',      type=int,   default=8,
                        help='batch size，显存不足时改为 4')
    parser.add_argument('--imgsz',      type=int,   default=640)
    parser.add_argument('--model',      default='yolov8n',
                        choices=['yolo11n', 'yolo11s', 'yolo11m'],
                        help='n=最快, s=均衡(推荐), m=最精准但慢')
    parser.add_argument('--model_path', default='',
                        help='本地权重路径(如 ./weights/yolo11s.pt)，设置后优先使用')
    parser.add_argument('--freeze',     type=int,   default=10,
                        help='冻结前N层(迁移学习)，数据少用10，数据多用0')
    parser.add_argument('--output_dir', default='./runs',
                        help='训练结果保存目录')
    parser.add_argument('--no_validate', action='store_true',
                        help='跳过训练后的验证步骤')
    args = parser.parse_args()

    # 检查依赖
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装: pip install ultralytics")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[warn] torch not found, will run on CPU (slow)")

    # 数据集统计
    print("\n数据集统计:")
    print_dataset_stats(args.data_dir)

    # 生成 YAML
    data_yaml = create_dataset_yaml(args.data_dir)

    # 加载预训练模型：优先 --model_path，其次 ./weights，最后自动下载
    if args.model_path:
        model_source = Path(args.model_path).expanduser().resolve()
        if not model_source.exists():
            raise FileNotFoundError(f"model_path not found: {model_source}")
        print(f"\n加载本地模型(--model_path): {model_source}")
    else:
        local_default = (Path(__file__).resolve().parent / 'weights' / f'{args.model}.pt').resolve()
        if local_default.exists():
            model_source = local_default
            print(f"\n加载本地模型(weights): {model_source}")
        else:
            model_source = f'{args.model}.pt'
            print(f"\n本地未找到 {local_default.name}，将使用 Ultralytics 自动下载: {model_source}")

    model = YOLO(str(model_source))

    # 打印训练配置
    train_kwargs = get_train_args(
        model_name  = args.model,
        data_yaml   = data_yaml,
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = args.imgsz,
        freeze      = args.freeze,
        output_dir  = args.output_dir,
    )

    print("\n训练配置:")
    for k, v in train_kwargs.items():
        print(f"  {k:20s}: {v}")

    # 开始训练
    print("\n" + "="*50)
    print(f"开始训练: {args.model} | epochs={args.epochs} | freeze={args.freeze}")
    print("="*50)
    t0 = time.time()

    results = model.train(**train_kwargs)

    elapsed = time.time() - t0
    print(f"\n训练完成，耗时: {elapsed/60:.1f} min")

    # 验证
    if not args.no_validate:
        validate_and_export(model, args.output_dir, args.model, data_yaml)

    # 打印最终结果路径
    best_pt = Path(args.output_dir) / f'{args.model}_screws' / 'weights' / 'best.pt'
    print(f"权重文件已保存到{best_pt}")


if __name__ == '__main__':
    main()
