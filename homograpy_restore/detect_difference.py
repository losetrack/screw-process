from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def compute_metrics(diff_gray: np.ndarray) -> Dict[str, float]:
    """Compute simple global difference metrics from a grayscale absolute diff map."""
    diff_float = diff_gray.astype(np.float32)
    mae = float(np.mean(diff_float))
    rmse = float(np.sqrt(np.mean(diff_float**2)))
    if rmse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(255.0 / rmse))
    return {"mae": mae, "rmse": rmse, "psnr": psnr}


def build_difference_mask(
    diff_gray: np.ndarray,
    threshold: int,
    morph_kernel: int,
    min_area: int,
) -> np.ndarray:
    """Build a clean binary mask for changed regions."""
    if threshold >= 0:
        _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        # Use Otsu to choose a threshold automatically.
        _, mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if min_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                filtered[labels == i] = 255
        mask = filtered

    return mask


def draw_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay changed regions in red on top of the input image."""
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(image_bgr, 0.75, overlay, 0.25, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    return out


def make_heatmap(diff_gray: np.ndarray) -> np.ndarray:
    """Convert absolute grayscale differences into a color heatmap."""
    return cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)


def compare_one(
    template_bgr: np.ndarray,
    target_bgr: np.ndarray,
    threshold: int,
    morph_kernel: int,
    min_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Compare one restored image against template and produce diagnostics."""
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    diff_gray = cv2.absdiff(template_gray, target_gray)
    mask = build_difference_mask(diff_gray, threshold, morph_kernel, min_area)
    heatmap = make_heatmap(diff_gray)
    overlay = draw_overlay(target_bgr, mask)

    metrics = compute_metrics(diff_gray)
    changed_pixels = int(cv2.countNonZero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    metrics["changed_pixels"] = float(changed_pixels)
    metrics["changed_ratio"] = float(changed_pixels / total_pixels) if total_pixels else 0.0

    return mask, heatmap, overlay, metrics


def process_all(
    template_path: Path,
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    threshold: int,
    morph_kernel: int,
    min_area: int,
    auto_resize: bool,
) -> None:
    template_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise FileNotFoundError(f"Template not found or unreadable: {template_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir} with pattern '{pattern}'")

    report_rows: List[Dict[str, str]] = []

    for img_path in files:
        target_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if target_bgr is None:
            print(f"[WARN] Skip unreadable image: {img_path.name}")
            continue

        if target_bgr.shape[:2] != template_bgr.shape[:2]:
            if not auto_resize:
                print(
                    f"[WARN] Skip {img_path.name}: size {target_bgr.shape[1]}x{target_bgr.shape[0]} "
                    f"!= template {template_bgr.shape[1]}x{template_bgr.shape[0]}"
                )
                continue
            target_bgr = cv2.resize(
                target_bgr,
                (template_bgr.shape[1], template_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        mask, heatmap, overlay, metrics = compare_one(
            template_bgr,
            target_bgr,
            threshold,
            morph_kernel,
            min_area,
        )

        stem = img_path.stem
        cv2.imwrite(str(output_dir / f"diff_mask_{stem}.png"), mask)
        cv2.imwrite(str(output_dir / f"diff_heatmap_{stem}.png"), heatmap)
        cv2.imwrite(str(output_dir / f"diff_overlay_{stem}.png"), overlay)

        row = {
            "image": img_path.name,
            "mae": f"{metrics['mae']:.4f}",
            "rmse": f"{metrics['rmse']:.4f}",
            "psnr": "inf" if np.isinf(metrics["psnr"]) else f"{metrics['psnr']:.4f}",
            "changed_pixels": f"{int(metrics['changed_pixels'])}",
            "changed_ratio": f"{metrics['changed_ratio']:.6f}",
        }
        report_rows.append(row)
        print(
            f"[OK] {img_path.name}: "
            f"changed_ratio={row['changed_ratio']}, mae={row['mae']}, psnr={row['psnr']}"
        )

    report_path = output_dir / "difference_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "mae", "rmse", "psnr", "changed_pixels", "changed_ratio"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Done. Report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect differences between restored images and a template.")
    parser.add_argument("--template", required=True, type=Path, help="Path to template image.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory of restored images to compare.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory for diff results.")
    parser.add_argument(
        "--pattern",
        default="raw_*_warp*.png",
        help="Glob pattern in input_dir (default: raw_*_warp*.png).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Binary threshold for absolute difference in [0,255]. Use -1 for Otsu.",
    )
    parser.add_argument(
        "--morph_kernel",
        type=int,
        default=3,
        help="Morphology kernel size for denoise (odd >=1, default: 3).",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=50,
        help="Minimum connected-component area in pixels (default: 50).",
    )
    parser.add_argument(
        "--auto_resize",
        action="store_true",
        help="Resize input image to template size when shape mismatch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.threshold < -1 or args.threshold > 255:
        raise ValueError("--threshold must be in [0,255] or -1 for Otsu")
    if args.morph_kernel < 1:
        raise ValueError("--morph_kernel must be >= 1")
    if args.morph_kernel % 2 == 0:
        args.morph_kernel += 1
    if args.min_area < 1:
        raise ValueError("--min_area must be >= 1")

    process_all(
        template_path=args.template,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        threshold=args.threshold,
        morph_kernel=args.morph_kernel,
        min_area=args.min_area,
        auto_resize=args.auto_resize,
    )


if __name__ == "__main__":
    main()
