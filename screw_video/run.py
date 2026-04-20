"""
Main entry point for video screw counting
Usage: python run.py --data_dir ./vedio_exp --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./masks
"""
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from detector import ScrewDetector
from tracker import ScrewTracker
from counter import ScrewCounter
from visualizer import draw_tracks


def process_video(
    video_path,
    detector,
    output_mask_dir,
    track_thresh,
    track_buffer,
    match_thresh,
    sample_interval=1,
    reid_weights_path=None,
    reid_model_name="osnet_x0_25",
    reid_match_thresh=0.75,
    reid_max_age=90,
    reid_device=None,
    edge_margin=0,
):
    """
    Process a single video: detect, track, count screws

    Args:
        video_path: Path to video file
        detector: ScrewDetector instance
        output_mask_dir: Directory to save mask visualization
        track_thresh: ByteTrack detection threshold
        track_buffer: ByteTrack buffer frames
        match_thresh: ByteTrack IOU threshold
        sample_interval: Keep one frame every N frames

    Returns:
        List of counts [Type_1, Type_2, Type_3, Type_4, Type_5]
    """
    print(f"\nProcessing {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return [0, 0, 0, 0, 0]

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    sampled_fps = max(1.0, fps / sample_interval)

    # Initialize tracker and counter with optimized parameters
    tracker = ScrewTracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=int(sampled_fps) if sampled_fps > 0 else 30,
        reid_weights_path=reid_weights_path,
        reid_model_name=reid_model_name,
        reid_match_thresh=reid_match_thresh,
        reid_max_age=reid_max_age,
        reid_device=reid_device,
        edge_margin=edge_margin,
    )
    counter = ScrewCounter()

    source_frame_idx = 0
    processed_frame_idx = 0
    mask_frame = None
    mask_tracks = None
    middle_frame_idx = total_frames // 2
    best_mask_distance = None
    expected_kept = (total_frames + sample_interval - 1) // sample_interval if total_frames > 0 else 0

    while cap.isOpened():
        grabbed = cap.grab()
        if not grabbed:
            break

        if source_frame_idx % sample_interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            # Detect screws in current frame
            detections = detector.detect(frame)

            # Update tracker
            tracks = tracker.update(detections, frame)

            # Update counter
            counter.update(tracks)

            # Save the tracked frame closest to the middle for mask visualization
            if len(tracks) > 0:
                distance_to_middle = abs(source_frame_idx - middle_frame_idx)
                if best_mask_distance is None or distance_to_middle < best_mask_distance:
                    best_mask_distance = distance_to_middle
                    mask_frame = frame.copy()
                    mask_tracks = tracks

            processed_frame_idx += 1

            if processed_frame_idx % 30 == 0:
                print(f"  Processed {processed_frame_idx}/{expected_kept} sampled frames...")

        source_frame_idx += 1

    cap.release()

    # Generate mask visualization
    if mask_frame is not None and mask_tracks is not None:
        mask_img = draw_tracks(mask_frame, mask_tracks)
        mask_path = output_mask_dir / f"{video_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask_img)
        print(f"  Saved mask to {mask_path}")

    # Get final counts using majority voting from tracker
    counts = counter.get_counts_with_voting(tracker)
    print(f"  Counts: {counts}")

    return counts


def main():
    parser = argparse.ArgumentParser(description='Video screw counting with YOLO + ByteTrack')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test videos')
    parser.add_argument('--output_path', type=str, default='./result.npy',
                        help='Output path for result.npy')
    parser.add_argument('--output_time_path', type=str, default='./time.txt',
                        help='Output path for time.txt')
    parser.add_argument('--mask_output_path', type=str, default='./mask_folder/',
                        help='Output directory for mask images')
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                        help='Path to YOLO weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.50,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--track_thresh', type=float, default=0.4,
                        help='ByteTrack: detection confidence threshold')
    parser.add_argument('--track_buffer', type=int, default=40,
                        help='ByteTrack: number of frames to keep lost tracks')
    parser.add_argument('--match_thresh', type=float, default=0.9,
                        help='ByteTrack: IOU matching threshold')
    parser.add_argument('--sample_interval', type=int, default=1,
                        help='When --process_sampled is enabled, keep one frame every N frames')
    parser.add_argument('--reid_weights', type=str, default=None,
                        help='Path to OSNet Re-ID weights; when provided, Re-ID-based ID matching is enabled')
    parser.add_argument('--reid_model', type=str, default='osnet_x0_25',
                        help='OSNet backbone variant for Re-ID')
    parser.add_argument('--reid_match_thresh', type=float, default=0.75,
                        help='Cosine similarity threshold for Re-ID ID matching')
    parser.add_argument('--reid_max_age', type=int, default=90,
                        help='Maximum frame gap for Re-ID matching')
    parser.add_argument('--reid_device', type=str, default=None,
                        help='Device for OSNet Re-ID model, e.g. cpu or cuda:0')
    parser.add_argument('--edge_margin', type=int, default=0,
                        help='Ignore detections whose boxes touch the image border within this many pixels')
    args = parser.parse_args()

    if args.sample_interval < 1:
        parser.error('--sample_interval must be >= 1')

    # Initialize detector
    print(f"Loading YOLO model from {args.weights}...")
    detector = ScrewDetector(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz
    )

    # Create output directories
    output_mask_dir = Path(args.mask_output_path)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files (deduplicated for case-insensitive file systems)
    data_dir = Path(args.data_dir)
    video_suffixes = {'.mp4', '.mov', '.avi'}
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

    if not video_files:
        print(f"No video files found in {data_dir}")
        return

    print(f"Found {len(video_files)} video(s)")

    # Process all videos
    start_time = time.time()
    results = {}

    for video_path in sorted(video_files):
        sample_interval = args.sample_interval
        counts = process_video(video_path, detector, output_mask_dir,
                              args.track_thresh, args.track_buffer, args.match_thresh,
                              sample_interval=sample_interval,
                              reid_weights_path=args.reid_weights,
                              reid_model_name=args.reid_model,
                              reid_match_thresh=args.reid_match_thresh,
                              reid_max_age=args.reid_max_age,
                              reid_device=args.reid_device,
                              edge_margin=args.edge_margin)
        results[video_path.stem] = counts

    total_time = time.time() - start_time

    # Save results
    print(f"\nSaving results to {args.output_path}...")
    np.save(args.output_path, results)

    print(f"Saving processing time to {args.output_time_path}...")
    with open(args.output_time_path, 'w') as f:
        f.write(f"{total_time:.2f}")

    print(f"\nDone! Total time: {total_time:.2f}s")
    print(f"Results: {results}")


if __name__ == '__main__':
    main()
