import argparse
import sys
from pathlib import Path

import cv2


def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(weights_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Please install: pip install ultralytics") from exc

    if not weights_path.exists():
        raise FileNotFoundError(f"weights file does not exist: {weights_path}")

    return YOLO(str(weights_path))


def main():
    parser = argparse.ArgumentParser(description="螺丝检测可视化推理脚本")
    parser.add_argument("--data_dir", required=True, help="测试图像文件夹路径")
    parser.add_argument("--weights", default="weights/best.pt", help="模型权重路径")
    parser.add_argument("--output_dir", default="./vis", help="可视化结果输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.30, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理尺寸")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    img_paths = sorted(
        list(data_dir.glob("*.jpg"))
        + list(data_dir.glob("*.jpeg"))
        + list(data_dir.glob("*.png"))
    )
    if not img_paths:
        print(f"[ERROR] No images found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = (Path(__file__).parent / weights_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path(__file__).parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights_path)
    device = get_device()

    print(f"Found {len(img_paths)} images")
    print(f"Model: {weights_path}")
    print(f"Output dir: {output_dir}")

    for img_path in img_paths:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )

        if not results:
            continue

        vis_img = results[0].plot()
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), vis_img)
        print(f"saved: {save_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
