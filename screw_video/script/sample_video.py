"""Sample images from videos.

Examples:
python sampler_video.py --video_path ./vedio_exp/IMG_2375.MOV --sample_interval 30
python sampler_video.py --data_dir ./vedio_exp --num_samples 50 --output_dir ./sampled_videos
"""

import argparse
from pathlib import Path

import cv2


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


def _validate_time_range(start_sec, end_sec):
	if start_sec < 0:
		raise ValueError("--start_sec must be >= 0")
	if end_sec is not None and end_sec <= start_sec:
		raise ValueError("--end_sec must be greater than --start_sec")


def _safe_read_video_meta(cap):
	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		fps = 30.0
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	return fps, total_frames


def _frame_range_from_time(fps, total_frames, start_sec, end_sec):
	start_frame = int(start_sec * fps)
	start_frame = max(0, min(start_frame, max(0, total_frames - 1)))

	if end_sec is None:
		end_frame = total_frames
	else:
		end_frame = int(end_sec * fps)
		end_frame = max(0, min(end_frame, total_frames))

	return start_frame, end_frame


def _save_frame(output_path, frame, image_ext, jpeg_quality):
	if image_ext.lower() in ["jpg", "jpeg"]:
		ok = cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
	else:
		ok = cv2.imwrite(str(output_path), frame)
	return ok


def sample_by_interval(
	cap,
	video_stem,
	out_dir,
	start_frame,
	end_frame,
	sample_interval,
	image_ext,
	jpeg_quality,
):
	"""Keep one frame every N frames in [start_frame, end_frame)."""
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

	saved_count = 0
	frame_idx = start_frame
	while frame_idx < end_frame:
		grabbed = cap.grab()
		if not grabbed:
			break

		if (frame_idx - start_frame) % sample_interval == 0:
			ok, frame = cap.retrieve()
			if not ok:
				break

			out_name = f"{video_stem}_f{frame_idx:06d}.{image_ext}"
			out_path = out_dir / out_name
			write_ok = _save_frame(out_path, frame, image_ext=image_ext, jpeg_quality=jpeg_quality)
			if write_ok:
				saved_count += 1

		frame_idx += 1

	return saved_count


def sample_by_count(
	cap,
	video_stem,
	out_dir,
	start_frame,
	end_frame,
	num_samples,
	image_ext,
	jpeg_quality,
):
	"""Uniformly sample a fixed number of frames in [start_frame, end_frame)."""
	frame_count = max(0, end_frame - start_frame)
	if frame_count == 0 or num_samples <= 0:
		return 0

	if num_samples == 1:
		target_frames = [start_frame]
	else:
		step = (frame_count - 1) / (num_samples - 1)
		target_frames = [int(round(start_frame + i * step)) for i in range(num_samples)]

	target_frames = sorted(set([f for f in target_frames if start_frame <= f < end_frame]))

	saved_count = 0
	for frame_idx in target_frames:
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
		ok, frame = cap.read()
		if not ok:
			continue

		out_name = f"{video_stem}_f{frame_idx:06d}.{image_ext}"
		out_path = out_dir / out_name
		write_ok = _save_frame(out_path, frame, image_ext=image_ext, jpeg_quality=jpeg_quality)
		if write_ok:
			saved_count += 1

	return saved_count


def sample_video(
	video_path,
	output_root,
	sample_interval,
	num_samples,
	start_sec,
	end_sec,
	image_ext,
	jpeg_quality,
):
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		print(f"[ERROR] Cannot open video: {video_path}")
		return 0

	fps, total_frames = _safe_read_video_meta(cap)
	start_frame, end_frame = _frame_range_from_time(
		fps=fps,
		total_frames=total_frames,
		start_sec=start_sec,
		end_sec=end_sec,
	)

	if end_frame <= start_frame:
		print(f"[WARN] Skip {video_path.name}: empty sampling range")
		cap.release()
		return 0

	out_dir = output_root / video_path.stem
	out_dir.mkdir(parents=True, exist_ok=True)

	print(
		f"Sampling {video_path.name} | FPS={fps:.2f}, Frames={total_frames}, "
		f"Range=[{start_frame}, {end_frame})"
	)

	if num_samples is not None:
		saved_count = sample_by_count(
			cap=cap,
			video_stem=video_path.stem,
			out_dir=out_dir,
			start_frame=start_frame,
			end_frame=end_frame,
			num_samples=num_samples,
			image_ext=image_ext,
			jpeg_quality=jpeg_quality,
		)
		mode_desc = f"num_samples={num_samples}"
	else:
		saved_count = sample_by_interval(
			cap=cap,
			video_stem=video_path.stem,
			out_dir=out_dir,
			start_frame=start_frame,
			end_frame=end_frame,
			sample_interval=sample_interval,
			image_ext=image_ext,
			jpeg_quality=jpeg_quality,
		)
		mode_desc = f"sample_interval={sample_interval}"

	cap.release()

	print(f"  Saved {saved_count} images ({mode_desc}) -> {out_dir}")
	return saved_count


def main():
	parser = argparse.ArgumentParser(description="Sample images from one or more videos")
	parser.add_argument("--video_path", type=str, default=None, help="Optional single video path")
	parser.add_argument("--data_dir", type=str, default="./vedio_exp", help="Directory containing videos")
	parser.add_argument("--output_dir", type=str, default="./sampled_videos", help="Output directory")

	mode_group = parser.add_mutually_exclusive_group()
	mode_group.add_argument("--sample_interval", type=int, default=30, help="Keep one frame every N frames")
	mode_group.add_argument("--num_samples", type=int, default=None, help="Uniformly sample K frames from each video")

	parser.add_argument("--start_sec", type=float, default=0.0, help="Start time in seconds")
	parser.add_argument("--end_sec", type=float, default=None, help="End time in seconds (exclusive)")
	parser.add_argument("--image_ext", type=str, default="jpg", choices=["jpg", "jpeg", "png"], help="Output image format")
	parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality (1-100)")
	args = parser.parse_args()

	if args.sample_interval is not None and args.sample_interval < 1:
		parser.error("--sample_interval must be >= 1")
	if args.num_samples is not None and args.num_samples < 1:
		parser.error("--num_samples must be >= 1")
	if not (1 <= args.jpeg_quality <= 100):
		parser.error("--jpeg_quality must be in [1, 100]")

	try:
		_validate_time_range(args.start_sec, args.end_sec)
	except ValueError as exc:
		parser.error(str(exc))

	if args.video_path:
		video_files = [Path(args.video_path)]
	else:
		video_files = find_video_files(Path(args.data_dir))

	if not video_files:
		print("No video files found.")
		return

	output_root = Path(args.output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	print(f"Found {len(video_files)} video(s)")
	total_saved = 0
	for video_path in video_files:
		saved = sample_video(
			video_path=video_path,
			output_root=output_root,
			sample_interval=args.sample_interval,
			num_samples=args.num_samples,
			start_sec=args.start_sec,
			end_sec=args.end_sec,
			image_ext=args.image_ext,
			jpeg_quality=args.jpeg_quality,
		)
		total_saved += saved

	print(f"Done. Total saved images: {total_saved}")


if __name__ == "__main__":
	main()
