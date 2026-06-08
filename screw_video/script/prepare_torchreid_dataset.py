"""
Convert the current Re-ID crop dataset into a torchreid-compatible Market1501 layout.

Input:
    reid_dataset/
        train/
            <identity_name>/
                *.jpg

Output:
    reid_dataset_torchreid/
        market1501/
            Market-1501-v15.09.15/
                bounding_box_train/
                query/
                bounding_box_test/
        split_summary.csv

Example:
    python prepare_torchreid_dataset.py
    python prepare_torchreid_dataset.py --input_dir ./reid_dataset/train --eval_ratio 0.2
"""
import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_identity_dirs(input_dir):
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def list_images(identity_dir):
    images = [path for path in identity_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images)


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_market1501_name(pid, camid, index, suffix):
    return f"{pid:04d}_c{camid}s1_{index:06d}{suffix.lower()}"


def assign_eval_split(images):
    """
    Split one identity's images into query and gallery.

    Since the current data is effectively single-camera, we assign pseudo camera ids:
    - query images use c1
    - gallery images use c2
    """
    if len(images) < 2:
        raise ValueError("Evaluation identity must contain at least 2 images")

    query_image = images[0]
    gallery_images = images[1:]
    return [query_image], gallery_images


def copy_with_new_name(src_path, dst_dir, file_name):
    dst_path = dst_dir / file_name
    shutil.copy2(src_path, dst_path)
    return dst_path


def convert_dataset(input_dir, output_root, eval_ratio, seed, min_images):
    market_root = output_root / "market1501"
    market_data_root = market_root / "Market-1501-v15.09.15"
    train_dir = market_data_root / "bounding_box_train"
    query_dir = market_data_root / "query"
    gallery_dir = market_data_root / "bounding_box_test"

    ensure_clean_dir(output_root)
    train_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    identity_dirs = list_identity_dirs(input_dir)
    usable = []
    dropped = []
    for identity_dir in identity_dirs:
        images = list_images(identity_dir)
        if len(images) < min_images:
            dropped.append((identity_dir.name, len(images)))
            continue
        usable.append((identity_dir.name, images))

    if len(usable) < 2:
        raise RuntimeError("Need at least 2 valid identities to build train/eval splits")

    rng = random.Random(seed)
    rng.shuffle(usable)

    eval_count = max(1, round(len(usable) * eval_ratio))
    eval_count = min(eval_count, len(usable) - 1)
    eval_set = usable[:eval_count]
    train_set = usable[eval_count:]

    split_rows = []
    train_image_count = 0
    query_image_count = 0
    gallery_image_count = 0

    for pid, (identity_name, images) in enumerate(train_set, start=1):
        for index, image_path in enumerate(images, start=1):
            camid = 1 + ((index - 1) % 6)
            new_name = make_market1501_name(pid, camid, index, image_path.suffix)
            copy_with_new_name(image_path, train_dir, new_name)
            split_rows.append({
                "original_identity": identity_name,
                "new_pid": pid,
                "subset": "train",
                "original_file": image_path.name,
                "new_file": new_name,
                "camid": camid,
            })
            train_image_count += 1

    eval_pid_start = len(train_set) + 1
    for pid_offset, (identity_name, images) in enumerate(eval_set):
        pid = eval_pid_start + pid_offset
        query_images, gallery_images = assign_eval_split(images)

        for index, image_path in enumerate(query_images, start=1):
            camid = 1
            new_name = make_market1501_name(pid, camid, index, image_path.suffix)
            copy_with_new_name(image_path, query_dir, new_name)
            split_rows.append({
                "original_identity": identity_name,
                "new_pid": pid,
                "subset": "query",
                "original_file": image_path.name,
                "new_file": new_name,
                "camid": camid,
            })
            query_image_count += 1

        for index, image_path in enumerate(gallery_images, start=1):
            camid = 2
            new_name = make_market1501_name(pid, camid, index, image_path.suffix)
            copy_with_new_name(image_path, gallery_dir, new_name)
            split_rows.append({
                "original_identity": identity_name,
                "new_pid": pid,
                "subset": "gallery",
                "original_file": image_path.name,
                "new_file": new_name,
                "camid": camid,
            })
            gallery_image_count += 1

    summary_path = output_root / "split_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_identity",
                "new_pid",
                "subset",
                "original_file",
                "new_file",
                "camid",
            ],
        )
        writer.writeheader()
        writer.writerows(split_rows)

    return {
        "usable_identities": len(usable),
        "dropped_identities": len(dropped),
        "train_identities": len(train_set),
        "eval_identities": len(eval_set),
        "train_images": train_image_count,
        "query_images": query_image_count,
        "gallery_images": gallery_image_count,
        "summary_path": summary_path,
        "dropped": dropped,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare a torchreid Market1501-style dataset")
    parser.add_argument("--input_dir", type=str, default="./reid_dataset/train", help="Current identity-folder dataset root")
    parser.add_argument("--output_dir", type=str, default="./reid_dataset_torchreid", help="Output root for torchreid dataset")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="Identity-level ratio used for query/gallery split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic identity split")
    parser.add_argument("--min_images", type=int, default=2, help="Minimum image count required to keep an identity")
    args = parser.parse_args()

    if not (0.0 < args.eval_ratio < 1.0):
        parser.error("--eval_ratio must be between 0 and 1")
    if args.min_images < 2:
        parser.error("--min_images must be >= 2")

    stats = convert_dataset(
        input_dir=Path(args.input_dir),
        output_root=Path(args.output_dir),
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_images=args.min_images,
    )

    print("Torchreid dataset prepared:")
    print(f"  usable identities: {stats['usable_identities']}")
    print(f"  dropped identities: {stats['dropped_identities']}")
    print(f"  train identities: {stats['train_identities']}")
    print(f"  eval identities: {stats['eval_identities']}")
    print(f"  train images: {stats['train_images']}")
    print(f"  query images: {stats['query_images']}")
    print(f"  gallery images: {stats['gallery_images']}")
    print(f"  split summary: {stats['summary_path']}")


if __name__ == "__main__":
    main()
