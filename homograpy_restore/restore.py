from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def create_color_frame_mask(
    image_bgr: np.ndarray,
    sat_threshold: int = 60,
    val_threshold: int = 50,
) -> np.ndarray:
    """Create a mask that keeps highly saturated marker colors and suppresses white background."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = np.where((saturation >= sat_threshold) & (value >= val_threshold), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def create_board_roi_from_color_frames(image_bgr: np.ndarray, pad: int = 15) -> np.ndarray:
    """Estimate the board ROI by taking the convex hull of colored marker pixels."""
    color_mask = create_color_frame_mask(image_bgr)
    ys, xs = np.where(color_mask > 0)
    roi = np.zeros(color_mask.shape, dtype=np.uint8)

    if len(xs) < 20:
        roi.fill(255)
        return roi

    points = np.column_stack((xs, ys)).astype(np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(roi, hull, 255)

    if pad > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))
        roi = cv2.dilate(roi, kernel, iterations=1)
    return roi


def create_screw_mask(
    image_bgr: np.ndarray,
    threshold: int = 230,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a mask using to cover the screw area
    255 represent the background (non-screw) area and 0 represents the screw area
    """
    # covert to gray image
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Threshold: pixels brighter than 215 are background
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operation optimization mask
    kernel = np.ones((5, 5), np.uint8)
    # Closing operation fills in the small black dots as noise in the background.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Outside ROI is ignored as background.
    if roi_mask is not None:
        mask = np.where(roi_mask > 0, mask, 255).astype(np.uint8)

    return mask


def flip_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Swap binary mask values strictly between 0 and 255."""
    return np.where(mask == 0, 255, 0).astype(np.uint8)


def extract_features(
    image_gray: np.ndarray,
    feature_extractor: cv2.Feature2D,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Tuple[cv2.KeyPoint, ...], Optional[np.ndarray]]:
    """Extract reusable local features for a grayscale image."""
    keypoints, descriptors = feature_extractor.detectAndCompute(image_gray, mask)
    if keypoints is None:
        return tuple(), descriptors
    return tuple(keypoints), descriptors


def extract_color_corner_features(
    image_bgr: np.ndarray,
    image_gray: np.ndarray,
    feature_extractor: cv2.Feature2D,
    max_corners: int = 2000,
    quality_level: float = 0.01,
    min_distance: float = 5.0,
    keypoint_size: float = 16.0,
) -> Tuple[Tuple[cv2.KeyPoint, ...], Optional[np.ndarray], np.ndarray]:
    """Use corners on colored frame regions as keypoints and compute descriptors there."""
    color_mask = create_color_frame_mask(image_bgr)
    corners = cv2.goodFeaturesToTrack(
        image_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=color_mask,
        blockSize=7,
        useHarrisDetector=False,
    )

    if corners is None:
        return tuple(), None, color_mask

    keypoints = [
        cv2.KeyPoint(float(pt[0][0]), float(pt[0][1]), keypoint_size)
        for pt in corners
    ]
    keypoints, descriptors = feature_extractor.compute(image_gray, keypoints)

    if keypoints is None:
        return tuple(), descriptors, color_mask
    return tuple(keypoints), descriptors, color_mask


def match_descriptors(
    desc_template: Optional[np.ndarray],
    desc_image: Optional[np.ndarray],
    matcher: cv2.BFMatcher,
    ratio_test: float,
) -> list:
    """Filter KNN matches using Lowe's ratio test."""
    if desc_template is None or desc_image is None:
        return []

    raw_matches = matcher.knnMatch(desc_template, desc_image, k=2)
    good_matches = []
    for m_n in raw_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)
    return good_matches


def is_quality_better(candidate_quality: Dict[str, Any], best_quality: Dict[str, Any]) -> bool:
    """Prefer reliable homographies first, then compare aggregate quality score."""
    better_reliability = candidate_quality["is_reliable"] and not best_quality["is_reliable"]
    better_score = candidate_quality["score"] > best_quality["score"]
    return better_reliability or better_score


def get_homography_method() -> int:
    """Use a more robust estimator when OpenCV provides it."""
    if hasattr(cv2, "USAC_MAGSAC"):
        return cv2.USAC_MAGSAC
    return cv2.RANSAC


def estimate_homography_from_matches(
    kps_template: Tuple[cv2.KeyPoint, ...],
    kps_image: Tuple[cv2.KeyPoint, ...],
    good_matches: list,
    ransac_thresh: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate the homography from image to template coordinates."""
    if len(good_matches) < 4:
        return None, None

    pts_template = np.float32([kps_template[m.queryIdx].pt for m in good_matches])
    pts_image = np.float32([kps_image[m.trainIdx].pt for m in good_matches])
    H, mask = cv2.findHomography(pts_image, pts_template, get_homography_method(), ransac_thresh)
    if H is None or mask is None:
        return None, None
    return H, mask


def _coverage_ratio(points: np.ndarray, image_shape: Tuple[int, int]) -> float:
    if len(points) == 0:
        return 0.0

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 0.0)
    width = max(float(image_shape[1] - 1), 1.0)
    height = max(float(image_shape[0] - 1), 1.0)
    return float((span[0] / width) * (span[1] / height))


def summarize_homography_quality(
    mask: Optional[np.ndarray],
    good_matches: list,
    kps_template: Tuple[cv2.KeyPoint, ...],
    kps_image: Tuple[cv2.KeyPoint, ...],
    template_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Measure whether the estimated homography is supported by enough, well-spread inliers."""
    inliers = int(mask.sum()) if mask is not None else 0
    total_matches = len(good_matches)
    inlier_ratio = (inliers / total_matches) if total_matches else 0.0

    coverage_template = 0.0
    coverage_image = 0.0
    if inliers > 0 and mask is not None:
        inlier_mask = mask.ravel().astype(bool)
        pts_template = np.float32([kps_template[m.queryIdx].pt for m in good_matches])[inlier_mask]
        pts_image = np.float32([kps_image[m.trainIdx].pt for m in good_matches])[inlier_mask]
        coverage_template = _coverage_ratio(pts_template, template_shape)
        coverage_image = _coverage_ratio(pts_image, image_shape)

    coverage = min(coverage_template, coverage_image)
    return {
        "inliers": inliers,
        "total_matches": total_matches,
        "inlier_ratio": inlier_ratio,
        "coverage": coverage,
        "is_reliable": inliers >= 18 and inlier_ratio >= 0.18 and coverage >= 0.12,
        "score": inliers * (0.5 + inlier_ratio) * (0.5 + coverage),
    }

def remove_extra_screws(
    restored: np.ndarray,
    template: np.ndarray,
    template_screw_mask: np.ndarray,
    dilation_kernel_size: int = 20,
    shadow_expand: int = 40,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove screws (and their shadows) that appear in restored image but not in the template.

    The restored image often contains a dark halo/shadow around screws. This function
    expands the detected screw mask in the restored image so the shadow region is also
    treated as part of the removed area

    Returns:
        A tuple of (restored_out, extra_screws_mask).
    """
    restored_screw_mask = create_screw_mask(restored, roi_mask=roi_mask)

    # Expand screw mask to cover surrounding shadow/halo
    if shadow_expand > 1:
        shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shadow_expand, shadow_expand))
        screw_region = flip_binary_mask(restored_screw_mask)
        screw_region = cv2.dilate(screw_region, shadow_kernel, iterations=1) 
        restored_screw_mask = flip_binary_mask(screw_region)
        
    extra_screws_mask = np.where(
        (template_screw_mask == 255) & (restored_screw_mask == 0),
        255,
        0,
    ).astype(np.uint8)

    if dilation_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        extra_screws_mask = cv2.dilate(extra_screws_mask, kernel, iterations=1)

    if cv2.countNonZero(extra_screws_mask) == 0:
        return restored.copy(), extra_screws_mask

    # distance-transform blending — fills screw cores with accurate template content.
    dist = cv2.distanceTransform(extra_screws_mask, cv2.DIST_L2, 5)
    blend_radius = max(3, dilation_kernel_size * 2)  # Wider blend zone
    alpha = np.clip(dist / float(blend_radius), 0.0, 1.0)
    alpha = np.power(alpha, 0.5)  # Harder blend (less gradual falloff)
    alpha = alpha[..., None]
    restored_out = (restored.astype(np.float32) * (1.0 - alpha) + template.astype(np.float32) * alpha).astype(np.uint8)

    return restored_out, extra_screws_mask

def find_homography_template_to_image(
    template_gray: np.ndarray,
    image_gray: np.ndarray,
    matcher: cv2.BFMatcher,
    feature_extractor: cv2.Feature2D,
    mask_template: Optional[np.ndarray] = None,
    mask_image: Optional[np.ndarray] = None, 
    ratio_test: float = 0.75,
    ransac_thresh: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[list]]:
    kps_template, desc_template = extract_features(template_gray, feature_extractor, mask_template)
    kps_image, desc_image = extract_features(image_gray, feature_extractor, mask_image)

    if desc_template is None or desc_image is None or len(kps_image) == 0:
        return None, None, kps_template, kps_image, []

    good_matches = match_descriptors(desc_template, desc_image, matcher, ratio_test)
    H, mask = estimate_homography_from_matches(kps_template, kps_image, good_matches, ransac_thresh)
    return H, mask, kps_template, kps_image, good_matches


def find_best_homography_template_to_image(
    template_gray: np.ndarray,
    image_gray: np.ndarray,
    matcher: cv2.BFMatcher,
    feature_extractor: cv2.Feature2D,
    mask_template: Optional[np.ndarray] = None,
    mask_image: Optional[np.ndarray] = None,
    ransac_thresh: float = 5.0,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Tuple[cv2.KeyPoint, ...],
    Tuple[cv2.KeyPoint, ...],
    list,
    Dict[str, Any],
]:
    """Try several matching settings and keep the most reliable result."""
    feature_sets = {
        "full": (
            extract_features(template_gray, feature_extractor, None),
            extract_features(image_gray, feature_extractor, None),
        )
    }
    if mask_template is not None and mask_image is not None:
        feature_sets["masked"] = (
            extract_features(template_gray, feature_extractor, mask_template),
            extract_features(image_gray, feature_extractor, mask_image),
        )

    candidate_settings = [
        ("masked", 0.65),
        ("full", 0.65),
        ("masked", 0.70),
        ("full", 0.70),
        ("full", 0.75),
    ]

    best_result = (None, None, tuple(), tuple(), [])
    best_quality: Dict[str, Any] = {
        "inliers": 0,
        "total_matches": 0,
        "inlier_ratio": 0.0,
        "coverage": 0.0,
        "is_reliable": False,
        "score": -1.0,
        "strategy": "none",
    }

    for feature_mode, ratio_test in candidate_settings:
        if feature_mode not in feature_sets:
            continue

        (kps_template, desc_template), (kps_image, desc_image) = feature_sets[feature_mode]
        if desc_template is None or desc_image is None or len(kps_image) == 0:
            continue

        good_matches = match_descriptors(desc_template, desc_image, matcher, ratio_test)
        H, mask = estimate_homography_from_matches(kps_template, kps_image, good_matches, ransac_thresh)
        quality = summarize_homography_quality(
            mask,
            good_matches,
            kps_template,
            kps_image,
            template_gray.shape,
            image_gray.shape,
        )
        quality["strategy"] = f"{feature_mode}_ratio_{ratio_test:.2f}"

        if is_quality_better(quality, best_quality):
            best_result = (H, mask, kps_template, kps_image, good_matches)
            best_quality = quality

    return (*best_result, best_quality)


def find_homography_from_color_corners(
    template_bgr: np.ndarray,
    template_gray: np.ndarray,
    image_bgr: np.ndarray,
    image_gray: np.ndarray,
    matcher: cv2.BFMatcher,
    feature_extractor: cv2.Feature2D,
    ransac_thresh: float = 5.0,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Tuple[cv2.KeyPoint, ...],
    Tuple[cv2.KeyPoint, ...],
    list,
    Dict[str, Any],
]:
    """Estimate homography by matching descriptors computed on colored-frame corners."""
    kps_template, desc_template, _ = extract_color_corner_features(
        template_bgr,
        template_gray,
        feature_extractor,
    )
    kps_image, desc_image, _ = extract_color_corner_features(
        image_bgr,
        image_gray,
        feature_extractor,
    )

    best_result = (None, None, tuple(), tuple(), [])
    best_quality: Dict[str, Any] = {
        "inliers": 0,
        "total_matches": 0,
        "inlier_ratio": 0.0,
        "coverage": 0.0,
        "is_reliable": False,
        "score": -1.0,
        "strategy": "color_corner_none",
    }

    if desc_template is None or desc_image is None or len(kps_template) == 0 or len(kps_image) == 0:
        return (*best_result, best_quality)

    for ratio_test in (0.65, 0.70, 0.75):
        good_matches = match_descriptors(desc_template, desc_image, matcher, ratio_test)
        H, mask = estimate_homography_from_matches(kps_template, kps_image, good_matches, ransac_thresh)
        quality = summarize_homography_quality(
            mask,
            good_matches,
            kps_template,
            kps_image,
            template_gray.shape,
            image_gray.shape,
        )
        quality["strategy"] = f"color_corner_ratio_{ratio_test:.2f}"

        if is_quality_better(quality, best_quality):
            best_result = (H, mask, kps_template, kps_image, good_matches)
            best_quality = quality

    return (*best_result, best_quality)

def warp_to_template(
    image: np.ndarray,
    H: np.ndarray,
    output_size: Tuple[int, int],
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """warp the input image to template image"""
    
    return cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR, borderValue=border_value)


def process_all(
    template_path: Path,
    input_dir: Path,
    output_dir: Path,
    pattern: str = "raw_*_warp*.png",
    debug: bool = False,
    remove_screws: bool = False,
) -> None:
    """Complete processing procedure"""
    
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_roi_mask = create_board_roi_from_color_frames(template)
    template_screw_mask = create_screw_mask(template, roi_mask=template_roi_mask)

    use_sift = hasattr(cv2, "SIFT_create")
    if use_sift:
        feature_extractor = cv2.SIFT_create(nfeatures=4000)
    else:
        feature_extractor = cv2.ORB_create(4000)
    norm_type = cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    akaze_extractor = cv2.AKAZE_create() if hasattr(cv2, "AKAZE_create") else None
    akaze_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False) if akaze_extractor is not None else None
    
    input_files = sorted(input_dir.glob(pattern))
    if not input_files:
        raise FileNotFoundError(f"No input images found in {input_dir} matching {pattern}")

    for in_path in input_files:
        print(f"Processing {in_path.name} ...")
        image = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [WARN] Could not read input: {in_path}")
            continue

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_roi_mask = create_board_roi_from_color_frames(image)

        # Generate screw mask 
        screw_mask = create_screw_mask(image, roi_mask=image_roi_mask)

        color_result = find_homography_from_color_corners(
            template,
            template_gray,
            image,
            image_gray,
            matcher,
            feature_extractor,
            ransac_thresh=5.0,
        )
        color_H, color_mask, color_kps_template, color_kps_image, color_good_matches, color_quality = color_result
        color_quality["strategy"] = (
            f"SIFT_{color_quality['strategy']}" if use_sift else f"ORB_{color_quality['strategy']}"
        )

        H, mask, kps_template, kps_image, good_matches, quality = find_best_homography_template_to_image(
            template_gray,
            image_gray,
            matcher,
            feature_extractor,
            mask_template=template_screw_mask,
            mask_image=screw_mask,
            ransac_thresh=5.0,
        )
        quality["strategy"] = f"SIFT_{quality['strategy']}" if use_sift else f"ORB_{quality['strategy']}"

        if is_quality_better(color_quality, quality):
            H = color_H
            mask = color_mask
            kps_template = color_kps_template
            kps_image = color_kps_image
            good_matches = color_good_matches
            quality = color_quality

        if (
            akaze_extractor is not None
            and akaze_matcher is not None
            and not quality["is_reliable"]
            and use_sift
        ):
            akaze_result = find_best_homography_template_to_image(
                template_gray,
                image_gray,
                akaze_matcher,
                akaze_extractor,
                mask_template=template_screw_mask,
                mask_image=screw_mask,
                ransac_thresh=5.0,
            )
            akaze_H, akaze_mask, akaze_kps_template, akaze_kps_image, akaze_good_matches, akaze_quality = akaze_result
            akaze_quality["strategy"] = f"AKAZE_{akaze_quality['strategy']}"
            quality["strategy"] = f"SIFT_{quality['strategy']}"
            if is_quality_better(akaze_quality, quality):
                H = akaze_H
                mask = akaze_mask
                kps_template = akaze_kps_template
                kps_image = akaze_kps_image
                good_matches = akaze_good_matches
                quality = akaze_quality

        if H is None:
            print(f"  [WARN] Homography estimation failed for {in_path.name}. Skipping.")
            continue

        if not quality["is_reliable"]:
            print(
                "  [WARN] Low-confidence homography for "
                f"{in_path.name}: strategy={quality['strategy']}, "
                f"inliers={quality['inliers']}, "
                f"inlier_ratio={quality['inlier_ratio']:.3f}, "
                f"coverage={quality['coverage']:.3f}"
            )

        restored = warp_to_template(image, H, (template.shape[1], template.shape[0]))
        extra_screws_mask = np.zeros(template_gray.shape, dtype=np.uint8)

        if remove_screws:
            # Use the screw mask on the template to remove the extra screws in the restored image
            restored, extra_screws_mask = remove_extra_screws(
                restored,
                template,
                template_screw_mask,
                dilation_kernel_size=15,
                roi_mask=template_roi_mask,
            )

        out_path = output_dir / in_path.name
        cv2.imwrite(str(out_path), restored)

        if debug:
            # Save the generated Mask in debug mode
            mask_debug_path = output_dir / f"mask_{in_path.name}"
            cv2.imwrite(str(mask_debug_path), screw_mask)

            # Save template screw mask
            
            template_mask_debug_path = output_dir / "mask_template.png"
            cv2.imwrite(str(template_mask_debug_path), template_screw_mask)

            template_roi_debug_path = output_dir / "roi_template.png"
            cv2.imwrite(str(template_roi_debug_path), template_roi_mask)

            roi_debug_path = output_dir / f"roi_{in_path.name}"
            cv2.imwrite(str(roi_debug_path), image_roi_mask)

            # Save the extra screws detected from the restored image
            extra_mask_debug_path = output_dir / f"extra_screws_{in_path.name}"
            cv2.imwrite(str(extra_mask_debug_path), extra_screws_mask)

            if good_matches is not None:
                vis = cv2.drawMatches(
                    template,
                    kps_template,
                    image,
                    kps_image,
                    good_matches[:200],
                    None,
                    matchColor=(0, 255, 0),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                debug_path = output_dir / f"debug_{in_path.name}"
                cv2.imwrite(str(debug_path), vis)

    print(f"Done. Restored images saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore warped images to a template using homography.")
    p.add_argument("--template", required=True, type=Path, help="Path to the template image.")
    p.add_argument("--input_dir", required=True, type=Path, help="Directory containing warped input images.")
    p.add_argument("--output_dir", required=True, type=Path, help="Directory to write restored images.")
    p.add_argument(
        "--pattern",
        default="raw_*_warp*.png",
        help="Glob pattern to match input images (default: raw_*_warp*.png).",
    )
    p.add_argument("--remove-screws", action="store_true", help="Remove extra screws")
    p.add_argument("--debug", action="store_true", help="Write debug match visualizations (slow).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_all(args.template, args.input_dir, args.output_dir, args.pattern, args.debug, args.remove_screws)


if __name__ == "__main__":
    main()
