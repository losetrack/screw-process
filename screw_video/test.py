"""
Visualization script for testing video screw counting algorithm.

Example:
python screw_video/test_visualize.py --data_dir ./screw_video/vedio_exp --output_dir ./screw_video/vis_videos --display
"""
import argparse
import time
from pathlib import Path

import cv2

from detector import ScrewDetector
from tracker import ScrewTracker
from counter import ScrewCounter
from visualizer import draw_tracks


def find_video_files(data_dir):
    """Collect video files with common extensions (deduplicated)."""
    video_suffixes = {".mp4", ".mov", ".avi"}
    video_files = []
    seen_paths = set()

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in video_suffixes:
            continue

        resolved_key = str(path.resolve()).lower()
        if resolved_key in seen_paths:
            continue

        seen_paths.add(resolved_key)
        video_files.append(path)

    return video_files


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
            1.8,
            (0, 0, 0),
            2,
        )
        y += 50


def process_video(
    video_path,
    detector,
    output_video_path,
    display=False,
    max_frames=-1,
    wait_ms=1,
    sample_interval=1,
    save_output_video=True,
    reid_weights_path=None,
    reid_model_name="osnet_x0_25",
    reid_match_thresh=0.75,
    reid_max_age=90,
    reid_device=None,
    edge_margin=0,
    min_track_length=1,
):
    """Run detection + tracking + counting and export visualization video."""
    print(f"Processing {video_path.name}...")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open video {video_path}")
        return [0, 0, 0, 0, 0], 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampled_fps = max(1.0, fps / sample_interval)
    print(
        f"  Total frames: {total_frames}, FPS: {fps:.2f}, "
        f"Sample interval: {sample_interval}, Size: {width}x{height}"
    )

    writer = None
    if save_output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, sampled_fps, (width, height))

    tracker = ScrewTracker(
        track_thresh=0.4,
        track_buffer=20,
        match_thresh=0.9,
        frame_rate=int(sampled_fps) if sampled_fps > 0 else 30,
        reid_weights_path=reid_weights_path,
        reid_model_name=reid_model_name,
        reid_match_thresh=reid_match_thresh,
        reid_max_age=reid_max_age,
        reid_device=reid_device,
        edge_margin=edge_margin,
    )
    counter = ScrewCounter(min_track_length=min_track_length)

    source_frame_idx = 0
    processed_frame_idx = 0
    expected_total = min(total_frames, max_frames) if max_frames > 0 else total_frames
    expected_kept = (expected_total + sample_interval - 1) // sample_interval if expected_total > 0 else 0

    while cap.isOpened():
        if max_frames > 0 and source_frame_idx >= max_frames:
            print(f"  Reached max_frames={max_frames}")
            break

        grabbed = cap.grab()
        if not grabbed:
            break

        if source_frame_idx % sample_interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            counter.update(tracks)

            vis_frame = draw_tracks(frame, tracks)
            draw_status_panel(
                vis_frame,
                frame_idx=source_frame_idx,
                det_count=len(detections),
                track_count=len(tracks),
                counts=counter.get_counts(),
            )

            if writer is not None:
                writer.write(vis_frame)

            if display:
                cv2.imshow("Screw Tracking Visualization", vis_frame)
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    print("  Stopped by user")
                    break

            processed_frame_idx += 1
            if processed_frame_idx % 30 == 0:
                print(f"  Processed {processed_frame_idx}/{expected_kept} sampled frames...")

        source_frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    counts = counter.get_counts_with_voting(tracker)
    if save_output_video:
        print(f"  Output video: {output_video_path}")
    else:
        print("  Output video: disabled")
    print(f"  Final counts: {counts}")
    return counts, processed_frame_idx, source_frame_idx


def main():
    parser = argparse.ArgumentParser(description="Visual test script for YOLO + ByteTrack screw counting")
    parser.add_argument("--data_dir", type=str, default="./vedio_exp", help="Directory containing test videos")
    parser.add_argument("--video_path", type=str, default=None, help="Optional single video path")
    parser.add_argument("--output_dir", type=str, default="./vis_videos", help="Directory for visualization videos")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit frames for quick debugging; -1 for full video")
    parser.add_argument("--display", action="store_true", help="Show live visualization window; press q to stop")
    parser.add_argument("--wait_ms", type=int, default=1, help="cv2.waitKey delay in milliseconds")
    parser.add_argument("--process_sampled", action="store_true", help="Run detection/tracking on sampled frames directly from source video")
    parser.add_argument("--sample_interval", type=int, default=1, help="When --process_sampled is enabled, keep one frame every N frames")
    parser.add_argument("--no_save_video", action="store_true", help="Disable writing visualization video to speed up testing")
    parser.add_argument("--reid_weights", type=str, default=None, help="Path to OSNet Re-ID weights; when provided, Re-ID-based ID matching is enabled")
    parser.add_argument("--reid_model", type=str, default="osnet_x0_25", help="OSNet backbone variant for Re-ID")
    parser.add_argument("--reid_match_thresh", type=float, default=0.85, help="Cosine similarity threshold for Re-ID ID matching")
    parser.add_argument("--reid_max_age", type=int, default=90, help="Maximum frame gap for Re-ID matching")
    parser.add_argument("--reid_device", type=str, default='cuda:0', help="Device for OSNet Re-ID model, e.g. cpu or cuda:0")
    parser.add_argument("--edge_margin", type=int, default=30, help="Ignore detections whose boxes touch the image border within this many pixels")
    parser.add_argument("--min_track_length", type=int, default=8, help="Ignore tracks shorter than this many associated frames when counting")
    args = parser.parse_args()

    if args.sample_interval < 1:
        parser.error("--sample_interval must be >= 1")

    if args.video_path:
        video_files = [Path(args.video_path)]
    else:
        video_files = find_video_files(Path(args.data_dir))

    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} video(s)")

    detector = ScrewDetector(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    results = {}
    frame_stats = {}

    for video_path in video_files:
        out_video = output_dir / f"{video_path.stem}_vis.mp4"

        if args.process_sampled:
            out_video = output_dir / f"{video_path.stem}_sampled_vis.mp4"

        if args.no_save_video:
            out_video = None

        counts, processed_frames, source_frames = process_video(
            video_path=video_path,
            detector=detector,
            output_video_path=out_video,
            display=args.display,
            max_frames=args.max_frames,
            wait_ms=args.wait_ms,
            sample_interval=args.sample_interval if args.process_sampled else 1,
            save_output_video=not args.no_save_video,
            reid_weights_path=args.reid_weights,
            reid_model_name=args.reid_model,
            reid_match_thresh=args.reid_match_thresh,
            reid_max_age=args.reid_max_age,
            reid_device=args.reid_device,
            edge_margin=args.edge_margin,
            min_track_length=args.min_track_length,
        )
        results[video_path.stem] = counts
        frame_stats[video_path.stem] = {
            "source_frames": source_frames,
            "processed_frames": processed_frames,
        }

    total_time = time.time() - start


    print("\nDone")
    print(f"Total time: {total_time:.2f}s")
    print(f"Frame stats: {frame_stats}")
    print(f"Counts: {results}")


if __name__ == "__main__":
    main()
