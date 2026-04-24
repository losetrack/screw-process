"""
Export Re-ID crops from videos using the current detection and tracking pipeline.

Example:
python export_reid_crops.py --data_dir ./vedio_exp --output_dir ./reid_dataset --min_track_length 3
python export_reid_crops.py --video_path ./vedio_exp/IMG_2374.MOV --reid_weights ./weights/osnet_x0_25_imagenet.pt
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2

from detector import ScrewDetector
from tracker import ScrewTracker


def find_video_files(data_dir):
    """Collect video files with common extensions (deduplicated)."""
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
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


def crop_box(frame, box):
    """Crop a detection box from the frame with boundary clamping."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = int(max(0, min(frame_w - 1, x1)))
    y1 = int(max(0, min(frame_h - 1, y1)))
    x2 = int(max(0, min(frame_w, x2)))
    y2 = int(max(0, min(frame_h, y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def remove_dir_if_exists(path):
    if path.exists():
        shutil.rmtree(path)


def finalize_video_tracks(video_path, tracker, track_records, temp_video_dir, train_dir, min_track_length, metadata_rows):
    """Move valid track folders into final dataset structure and drop short tracks."""
    kept_tracks = 0
    removed_tracks = 0

    for track_id, record in sorted(track_records.items()):
        track_length = record["length"]
        temp_track_dir = record["dir"]

        if track_length < min_track_length:
            remove_dir_if_exists(temp_track_dir)
            removed_tracks += 1
            continue

        final_class = tracker.get_track_final_class(track_id)
        identity_name = f"{video_path.stem}_tid{track_id:04d}"
        final_track_dir = train_dir / identity_name
        remove_dir_if_exists(final_track_dir)
        final_track_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_track_dir), str(final_track_dir))

        metadata_rows.append({
            "identity": identity_name,
            "video": video_path.stem,
            "track_id": track_id,
            "track_length": track_length,
            "final_class": final_class if final_class is not None else -1,
            "crop_dir": str(final_track_dir),
        })
        kept_tracks += 1

    remove_dir_if_exists(temp_video_dir)
    return kept_tracks, removed_tracks


def export_video_crops(
    video_path,
    detector,
    output_root,
    track_thresh,
    track_buffer,
    match_thresh,
    sample_interval,
    reid_weights_path,
    reid_model_name,
    reid_match_thresh,
    reid_max_age,
    reid_device,
    edge_margin,
    min_track_length,
    image_ext,
    jpeg_quality,
    metadata_rows,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sampled_fps = max(1.0, fps / sample_interval) if fps > 0 else 30.0

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

    temp_root = output_root / "_tmp"
    temp_video_dir = temp_root / video_path.stem
    train_dir = output_root / "train"
    remove_dir_if_exists(temp_video_dir)
    temp_video_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    track_records = {}
    source_frame_idx = 0
    processed_frame_idx = 0
    expected_kept = (total_frames + sample_interval - 1) // sample_interval if total_frames > 0 else 0

    print(
        f"Exporting {video_path.name} | frames={total_frames}, fps={fps:.2f}, "
        f"sample_interval={sample_interval}"
    )

    while cap.isOpened():
        grabbed = cap.grab()
        if not grabbed:
            break

        if source_frame_idx % sample_interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)

            for track in tracks:
                track_id = track["track_id"]
                class_id = track["class"]
                crop = crop_box(frame, track["box"])
                if crop is None:
                    continue

                if track_id not in track_records:
                    track_dir = temp_video_dir / f"track_{track_id:04d}"
                    track_dir.mkdir(parents=True, exist_ok=True)
                    track_records[track_id] = {
                        "dir": track_dir,
                        "length": 0,
                        "first_class": class_id,
                    }

                track_records[track_id]["length"] += 1
                track_dir = track_records[track_id]["dir"]
                image_name = (
                    f"{video_path.stem}_tid{track_id:04d}_"
                    f"f{source_frame_idx:06d}_c{class_id}.{image_ext}"
                )
                image_path = track_dir / image_name

                if image_ext.lower() in {"jpg", "jpeg"}:
                    cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                else:
                    cv2.imwrite(str(image_path), crop)

            processed_frame_idx += 1
            if processed_frame_idx % 30 == 0:
                print(f"  Processed {processed_frame_idx}/{expected_kept} sampled frames...")

        source_frame_idx += 1

    cap.release()

    kept_tracks, removed_tracks = finalize_video_tracks(
        video_path=video_path,
        tracker=tracker,
        track_records=track_records,
        temp_video_dir=temp_video_dir,
        train_dir=train_dir,
        min_track_length=min_track_length,
        metadata_rows=metadata_rows,
    )
    print(f"  Kept tracks: {kept_tracks}, removed short tracks: {removed_tracks}")
    return kept_tracks, removed_tracks


def write_metadata(output_root, metadata_rows):
    metadata_path = output_root / "metadata.csv"
    fieldnames = ["identity", "video", "track_id", "track_length", "final_class", "crop_dir"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Re-ID crops from videos using the current pipeline")
    parser.add_argument("--video_path", type=str, default=None, help="Optional single video path")
    parser.add_argument("--data_dir", type=str, default="./vedio_exp", help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, default="./reid_dataset", help="Output dataset root")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--track_thresh", type=float, default=0.4, help="ByteTrack detection confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=40, help="ByteTrack lost-track buffer")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="ByteTrack IoU matching threshold")
    parser.add_argument("--sample_interval", type=int, default=1, help="Keep one frame every N frames")
    parser.add_argument("--reid_weights", type=str, default=None, help="Optional OSNet Re-ID weights")
    parser.add_argument("--reid_model", type=str, default="osnet_x0_25", help="OSNet backbone variant")
    parser.add_argument("--reid_match_thresh", type=float, default=0.85, help="Cosine similarity threshold for Re-ID")
    parser.add_argument("--reid_max_age", type=int, default=90, help="Maximum frame gap for Re-ID matching")
    parser.add_argument("--reid_device", type=str, default=None, help="Device for OSNet Re-ID model")
    parser.add_argument("--edge_margin", type=int, default=30, help="Ignore detections close to image border")
    parser.add_argument("--min_track_length", type=int, default=5, help="Minimum track length to keep")
    parser.add_argument("--image_ext", type=str, default="jpg", choices=["jpg", "jpeg", "png"], help="Output image format")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality (1-100)")
    args = parser.parse_args()

    if args.sample_interval < 1:
        parser.error("--sample_interval must be >= 1")
    if args.min_track_length < 1:
        parser.error("--min_track_length must be >= 1")
    if not (1 <= args.jpeg_quality <= 100):
        parser.error("--jpeg_quality must be in [1, 100]")

    if args.video_path:
        video_files = [Path(args.video_path)]
    else:
        video_files = find_video_files(Path(args.data_dir))

    if not video_files:
        print("No video files found.")
        return

    detector = ScrewDetector(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(video_files)} video(s)")
    metadata_rows = []
    total_kept = 0
    total_removed = 0

    for video_path in video_files:
        kept_tracks, removed_tracks = export_video_crops(
            video_path=video_path,
            detector=detector,
            output_root=output_root,
            track_thresh=args.track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            sample_interval=args.sample_interval,
            reid_weights_path=args.reid_weights,
            reid_model_name=args.reid_model,
            reid_match_thresh=args.reid_match_thresh,
            reid_max_age=args.reid_max_age,
            reid_device=args.reid_device,
            edge_margin=args.edge_margin,
            min_track_length=args.min_track_length,
            image_ext=args.image_ext,
            jpeg_quality=args.jpeg_quality,
            metadata_rows=metadata_rows,
        )
        total_kept += kept_tracks
        total_removed += removed_tracks

    write_metadata(output_root, metadata_rows)
    print(f"Done. Kept tracks: {total_kept}, removed short tracks: {total_removed}")


if __name__ == "__main__":
    main()
