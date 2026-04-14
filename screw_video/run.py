"""
run.py — Lab4 视频螺丝计数提交入口

用法:
    python run.py \
        --data_dir   /path/to/videos \
        --output_path      ./result.npy \
        --output_time_path ./time.txt \
        --mask_output_path ./masks/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="视频螺丝计数")
    p.add_argument("--data_dir",          required=True,
                   help="包含测试视频的文件夹路径")
    p.add_argument("--output_path",       default="./result.npy",
                   help="输出 .npy 文件路径")
    p.add_argument("--output_time_path",  default="./time.txt",
                   help="输出时间记录文件路径")
    p.add_argument("--mask_output_path",  default="./masks/",
                   help="掩膜图像输出文件夹路径")
    p.add_argument("--weights",           default="weights/best.pt",
                   help="YOLO 权重路径（相对于 run.py 所在目录）")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--frame_step", type=int, default=5,
                   help="每隔 N 帧采样一次（默认 5）")
    p.add_argument("--dist_thresh", type=float, default=50.0,
                   help="GlobalMap 去重距离阈值（像素，默认 50）")
    p.add_argument("--aligner", default="sift", choices=["sift", "orb"],
                   help="帧间对齐特征提取器（默认 sift）")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 路径解析 ──────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    video_paths = sorted(
        p for p in data_dir.iterdir()
        if p.suffix.lower() in video_exts
    )
    if not video_paths:
        print(f"[ERROR] No video files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = Path(__file__).parent / weights_path

    mask_dir = Path(args.mask_output_path)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(video_paths)} video(s) in {data_dir}")
    print(f"Weights : {weights_path}")
    print(f"frame_step={args.frame_step}, dist_thresh={args.dist_thresh}, "
          f"aligner={args.aligner}")

    # ── 初始化处理器（模型只加载一次）────────────────────────────────
    # Import here so the module is found whether run from project root or screw_video/
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from screw_video.video_processor import VideoProcessor

    processor = VideoProcessor(
        weights_path=weights_path,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        frame_step=args.frame_step,
        dist_thresh=args.dist_thresh,
        aligner_method=args.aligner,
    )

    # ── 逐视频处理 ────────────────────────────────────────────────────
    import cv2

    out_dict = {}
    t_start = time.time()

    for vpath in video_paths:
        print(f"\nProcessing: {vpath.name}")
        t0 = time.time()

        try:
            counts, mask_frame = processor.process(vpath)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            counts = [0] * 5
            mask_frame = None

        elapsed = time.time() - t0
        key = vpath.stem
        out_dict[key] = counts
        print(f"  counts={counts}  ({elapsed:.1f}s)")

        # 保存掩膜图像
        if mask_frame is not None:
            mask_path = mask_dir / f"{key}_mask.png"
            cv2.imwrite(str(mask_path), mask_frame)
            print(f"  mask → {mask_path}")

    total_time = time.time() - t_start

    # ── 保存结果 ──────────────────────────────────────────────────────
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), out_dict)

    time_path = Path(args.output_time_path)
    time_path.parent.mkdir(parents=True, exist_ok=True)
    time_path.write_text(str(total_time))

    # ── 格式验证 ──────────────────────────────────────────────────────
    loaded = np.load(str(output_path), allow_pickle=True).item()
    assert isinstance(loaded, dict), "输出格式错误: 必须是字典"
    for k, v in loaded.items():
        assert len(v) == 5, f"格式错误: {k} 的 value 长度不为 5"

    print(f"\n{'='*50}")
    print(f"Videos processed : {len(video_paths)}")
    print(f"Total time       : {total_time:.2f}s")
    print(f"Result saved to  : {output_path}")
    print(f"Time saved to    : {time_path}")
    print(f"Masks saved to   : {mask_dir}")
    print("Output format check: PASSED")


if __name__ == "__main__":
    main()
