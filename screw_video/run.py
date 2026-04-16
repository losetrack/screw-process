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


def process_video(video_path, detector, output_mask_dir, track_thresh, track_buffer, match_thresh):
    """
    Process a single video: detect, track, count screws

    Args:
        video_path: Path to video file
        detector: ScrewDetector instance
        output_mask_dir: Directory to save mask visualization
        track_thresh: ByteTrack detection threshold
        track_buffer: ByteTrack buffer frames
        match_thresh: ByteTrack IOU threshold

    Returns:
        List of counts [Type_1, Type_2, Type_3, Type_4, Type_5]
    """
    print(f"Processing {video_path.name}...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return [0, 0, 0, 0, 0]

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")

    # Initialize tracker and counter with optimized parameters
    tracker = ScrewTracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=int(fps) if fps > 0 else 30
    )
    counter = ScrewCounter()

    frame_idx = 0
    mask_frame = None
    mask_tracks = None
    middle_frame_idx = total_frames // 2
    best_mask_distance = None

    while cap.isOpened():
        ret, frame = cap.read()
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
            distance_to_middle = abs(frame_idx - middle_frame_idx)
            if best_mask_distance is None or distance_to_middle < best_mask_distance:
                best_mask_distance = distance_to_middle
                mask_frame = frame.copy()
                mask_tracks = tracks

        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

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
    parser.add_argument('--track_thresh', type=float, default=0.35,
                        help='ByteTrack: detection confidence threshold')
    parser.add_argument('--track_buffer', type=int, default=45,
                        help='ByteTrack: number of frames to keep lost tracks')
    parser.add_argument('--match_thresh', type=float, default=0.85,
                        help='ByteTrack: IOU matching threshold')
    args = parser.parse_args()

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

    # Find all video files
    data_dir = Path(args.data_dir)
    video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']
    video_files = []
    for ext in video_extensions:
        video_files.extend(data_dir.glob(ext))

    if not video_files:
        print(f"No video files found in {data_dir}")
        return

    print(f"Found {len(video_files)} video(s)")

    # Process all videos
    start_time = time.time()
    results = {}

    for video_path in sorted(video_files):
        counts = process_video(video_path, detector, output_mask_dir,
                              args.track_thresh, args.track_buffer, args.match_thresh)
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
