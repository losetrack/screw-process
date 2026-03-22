"""
run.py - 作业提交入口
用法:
    python run.py --data_dir /path/to/test_images \
                  --output_path ./result.npy \
                  --output_time_path ./time.txt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def load_model(weights_path):
    """加载 YOLO 模型"""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install: pip install ultralytics")

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"weights file does not exists: {weights_path}")

    model = YOLO(weights_path)
    print(f"Model loaded: {weights_path}")
    return model


def predict_image(model, img_path, conf_thresh=0.25, iou_thresh=0.45, num_classes=5):
    """
    对单张图推理，返回 5 个类别的计数列表。
    """
    results = model.predict(
        source    = str(img_path),
        conf      = conf_thresh,
        iou       = iou_thresh,
        imgsz     = 640,
        verbose   = False,
        device    = _get_device(),
    )

    counts = [0] * num_classes
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls.item())
            if 0 <= cls < num_classes:
                counts[cls] += 1

    return counts


def _get_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='螺丝计数推理脚本')
    parser.add_argument('--data_dir',        required=True,
                        help='测试图像文件夹路径')
    parser.add_argument('--output_path',     default='./result.npy',
                        help='输出 .npy 文件路径')
    parser.add_argument('--output_time_path',default='./time.txt',
                        help='输出时间记录文件路径')
    parser.add_argument('--weights',         default='weights/best.pt',
                        help='YOLOv8 权重文件路径')
    parser.add_argument('--conf',  type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou',   type=float, default=0.45, help='NMS IoU 阈值')
    args = parser.parse_args()

    # 检查输入目录 
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    img_paths = sorted(
        list(data_dir.glob('*.jpg')) +
        list(data_dir.glob('*.jpeg')) +
        list(data_dir.glob('*.png'))
    )
    if not img_paths:
        print(f"[ERROR] No images found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(img_paths)} images in {data_dir}")

    # 加载模型 
    # weights 路径支持相对于 run.py 所在目录
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = Path(__file__).parent / weights_path

    model = load_model(weights_path)

    # 推理 
    out_dict = {}
    t_start  = time.time()

    for img_path in img_paths:
        t_img = time.time()
        counts = predict_image(model, img_path,
                               conf_thresh=args.conf,
                               iou_thresh=args.iou)
        elapsed_img = time.time() - t_img

        key = img_path.stem          # 不带后缀的文件名
        out_dict[key] = counts
        print(f"  {key}: {counts}  ({elapsed_img:.2f}s)")

    total_time = time.time() - t_start

    # 保存结果 
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), out_dict)

    time_path = Path(args.output_time_path)
    time_path.parent.mkdir(parents=True, exist_ok=True)
    time_path.write_text(str(total_time))

    # 打印汇总
    print(f"\n{'='*50}")
    print(f"Total images processed : {len(img_paths)}")
    print(f"Total processing time  : {total_time:.2f}s")
    print(f"Average per image      : {total_time/len(img_paths):.2f}s")
    print(f"Result saved to        : {output_path}")
    print(f"Time saved to          : {time_path}")
    print(f"{'='*50}")

    # 快速验证格式
    loaded = np.load(str(output_path), allow_pickle=True).item()
    assert isinstance(loaded, dict), "输出格式错误: 必须是字典"
    for k, v in loaded.items():
        assert len(v) == 5, f"格式错误: {k} 的 value 长度不为 5"
    print("Output format check: PASSED")


if __name__ == '__main__':
    main()
