from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

def create_screw_mask(image_bgr: np.ndarray, threshold: int = 230) -> np.ndarray:
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

    return mask


def flip_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Swap binary mask values strictly between 0 and 255."""
    return np.where(mask == 0, 255, 0).astype(np.uint8)

def remove_extra_screws(
    restored: np.ndarray,
    template: np.ndarray,
    template_screw_mask: np.ndarray,
    dilation_kernel_size: int = 20,
    shadow_expand: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove screws (and their shadows) that appear in restored image but not in the template.

    The restored image often contains a dark halo/shadow around screws. This function
    expands the detected screw mask in the restored image so the shadow region is also
    treated as part of the removed area

    Returns:
        A tuple of (restored_out, extra_screws_mask).
    """
    restored_screw_mask = create_screw_mask(restored)

    # Expand screw mask to cover surrounding shadow/halo
    if shadow_expand > 1:
        shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shadow_expand, shadow_expand))
        screw_region = flip_binary_mask(restored_screw_mask)
        screw_region = cv2.dilate(screw_region, shadow_kernel, iterations=1) 
        restored_screw_mask = flip_binary_mask(screw_region)
        
    # template_screw_mask == 255 : non-screw area in template
    # restored_screw_mask == 0 : screw area in restored image
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
    
    # template extract Overall feature of the image 
    kps_template, desc_template = feature_extractor.detectAndCompute(template_gray, mask_template)
    kps_image, desc_image = feature_extractor.detectAndCompute(image_gray, mask_image)

    if desc_template is None or desc_image is None or len(kps_image) == 0:
        return None, None, kps_template, kps_image, []

    # Match descriptors with KNN and apply ratio test
    raw_matches = matcher.knnMatch(desc_template, desc_image, k=2)
    good_matches = []
    for m_n in raw_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None, None, kps_template, kps_image, good_matches

    pts_template = np.float32([kps_template[m.queryIdx].pt for m in good_matches])
    pts_image = np.float32([kps_image[m.trainIdx].pt for m in good_matches])

    H, mask = cv2.findHomography(pts_image, pts_template, cv2.RANSAC, ransac_thresh)
    
    if H is None or mask is None:
        return None, None, kps_template, kps_image, good_matches
    
    return H, mask, kps_template, kps_image, good_matches

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
    template_screw_mask = create_screw_mask(template)

    use_sift = hasattr(cv2, "SIFT_create")
    if use_sift:
        feature_extractor = cv2.SIFT_create(nfeatures=5000)
    else:
        feature_extractor = cv2.ORB_create(5000)
    norm_type = cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    
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
        
        # Generate screw mask 
        screw_mask = create_screw_mask(image)

        # input mask_template + mask_image
        H, mask, kps_template, kps_image, good_matches = find_homography_template_to_image(
            template_gray,
            image_gray,
            matcher,
            feature_extractor,
            mask_image=screw_mask,
            mask_template=template_screw_mask,
            ratio_test=0.70,
            ransac_thresh=5.0,
        )

        if H is None:
            print(f"  [WARN] Homography estimation failed for {in_path.name}. Skipping.")
            continue

        restored = warp_to_template(image, H, (template.shape[1], template.shape[0]))

        if remove_screws:
            # Use the screw mask on the template to remove the extra screws in the restored image
            restored, extra_screws_mask = remove_extra_screws(restored, template, template_screw_mask, dilation_kernel_size=15)

        out_path = output_dir / in_path.name
        cv2.imwrite(str(out_path), restored)

        if debug:
            # Save the generated Mask in debug mode
            mask_debug_path = output_dir / f"mask_{in_path.name}"
            cv2.imwrite(str(mask_debug_path), screw_mask)

            # Save template screw mask
            
            template_mask_debug_path = output_dir / "mask_template.png"
            cv2.imwrite(str(template_mask_debug_path), template_screw_mask)

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
