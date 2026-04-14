"""
Visualization script for testing video screw counting algorithm.

Example:
python screw_video/test_visualize.py --data_dir ./screw_video/vedio_exp --output_dir ./screw_video/vis_videos --display
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector import ScrewDetector
from tracker import ScrewTracker
from counter import ScrewCounter
from visualizer import draw_tracks


def find_video_files(data_dir):
    """Collect video files with common extensions."""
    video_extensions = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(data_dir.glob(ext))
    return sorted(video_files)


def draw_status_panel(frame, frame_idx, det_count, track_count, counts):
    """Overlay status text for quick algorithm debugging."""
    info = [
        f"Frame: {frame_idx}",
        f"Detections: {det_count}",
        f"Active Tracks: {track_count}",
        "Counts: " + ", ".join([f"T{i + 1}:{c}" for i, c in enumerate(counts)]),
    ]

    y = 24
    for text in info:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        y += 26


def process_video(video_path, detector, output_video_path, display=False, max_frames=-1, wait_ms=1):
    """Run detection + tracking + counting and export visualization video."""
    print(f"Processing {video_path.name}...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open video {video_path}")
        return [0, 0, 0, 0, 0], 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}, Size: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    tracker = ScrewTracker(
        track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=int(fps),
    )
    counter = ScrewCounter()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        counter.update(tracks)

        vis_frame = draw_tracks(frame, tracks)
        draw_status_panel(
            vis_frame,
            frame_idx=frame_idx,
            det_count=len(detections),
            track_count=len(tracks),
            counts=counter.get_counts(),
        )

        writer.write(vis_frame)

        if display:
            cv2.imshow("Screw Tracking Visualization", vis_frame)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                print("  Stopped by user")
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

        if max_frames > 0 and frame_idx >= max_frames:
            print(f"  Reached max_frames={max_frames}")
            break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    counts = counter.get_counts()
    print(f"  Output video: {output_video_path}")
    print(f"  Final counts: {counts}")
    return counts, frame_idx


def main():
    parser = argparse.ArgumentParser(description="Visual test script for YOLO + ByteTrack screw counting")
    parser.add_argument("--data_dir", type=str, default="./vedio_exp", help="Directory containing test videos")
    parser.add_argument("--video_path", type=str, default=None, help="Optional single video path")
    parser.add_argument("--output_dir", type=str, default="./vis_videos", help="Directory for visualization videos")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit frames for quick debugging; -1 for full video")
    parser.add_argument("--display", action="store_true", help="Show live visualization window; press q to stop")
    parser.add_argument("--wait_ms", type=int, default=1, help="cv2.waitKey delay in milliseconds")
    args = parser.parse_args()

    detector = ScrewDetector(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video_path:
        video_files = [Path(args.video_path)]
    else:
        video_files = find_video_files(Path(args.data_dir))

    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} video(s)")

    start = time.time()
    results = {}
    frame_stats = {}

    for video_path in video_files:
        out_video = output_dir / f"{video_path.stem}_vis.mp4"
        counts, processed_frames = process_video(
            video_path=video_path,
            detector=detector,
            output_video_path=out_video,
            display=args.display,
            max_frames=args.max_frames,
            wait_ms=args.wait_ms,
        )
        results[video_path.stem] = counts
        frame_stats[video_path.stem] = processed_frames

    total_time = time.time() - start


    print("\nDone")
    print(f"Total time: {total_time:.2f}s")
    print(f"Frame stats: {frame_stats}")
    print(f"Counts: {results}")


if __name__ == "__main__":
    main()
