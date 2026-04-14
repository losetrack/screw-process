"""
FrameAligner: estimates homography from video frames to a fixed reference frame.

Adapted from homograpy_restore/restore.py's feature matching pipeline.
Used by global_map.py to project per-frame detections into a common coordinate space.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class FrameAligner:
    """
    Estimates H that maps points in a given frame to the reference frame's coordinate space.

    Two-strategy approach (in order):
      1. Direct:      match current frame → reference frame  (no drift, preferred)
      2. Incremental: match current frame → previous frame, compose with H_prev→ref
                      (handles large camera displacement between current and reference)

    If both fail, the last known H is reused (with a fallback counter incremented).
    Call set_reference() once per video, then align() for each sampled frame.
    Call reset() between videos.
    """

    def __init__(
        self,
        method: str = "sift",
        ratio_test: float = 0.70,
        ransac_thresh: float = 5.0,
        min_inliers: int = 10,
        max_features: int = 3000,
    ):
        """
        Args:
            method:        Feature extractor — 'sift' (default, more accurate) or 'orb' (faster).
                           Falls back to ORB automatically if SIFT is unavailable.
            ratio_test:    Lowe's ratio test threshold for descriptor matching.
            ransac_thresh: RANSAC reprojection error threshold (pixels).
            min_inliers:   Minimum inlier count to accept a homography estimate.
            max_features:  Maximum keypoints to detect per frame.
        """
        if method == "sift" and hasattr(cv2, "SIFT_create"):
            self._extractor = cv2.SIFT_create(nfeatures=max_features)
            self._norm = cv2.NORM_L2
            self.method = "sift"
        else:
            self._extractor = cv2.ORB_create(nfeatures=max_features)
            self._norm = cv2.NORM_HAMMING
            self.method = "orb"

        self.ratio_test = ratio_test
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers

        self._matcher = cv2.BFMatcher(normType=self._norm, crossCheck=False)

        # Reference frame features
        self._ref_kps: list = []
        self._ref_desc: Optional[np.ndarray] = None

        # Previous frame features + cumulative H (prev → ref)
        self._prev_kps: list = []
        self._prev_desc: Optional[np.ndarray] = None
        self._H_prev_to_ref: Optional[np.ndarray] = None

        # Diagnostics
        self.stats: dict = {"direct": 0, "incremental": 0, "fallback": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, frame: np.ndarray) -> None:
        """
        Designate `frame` as the coordinate origin for all subsequent align() calls.
        Must be called once before align().
        """
        gray = _to_gray(frame)
        self._ref_kps, self._ref_desc = self._detect(gray)
        # Initialise previous-frame state to the reference itself
        self._prev_kps = self._ref_kps
        self._prev_desc = self._ref_desc
        self._H_prev_to_ref = np.eye(3, dtype=np.float64)
        self.stats = {"direct": 0, "incremental": 0, "fallback": 0}

    def align(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate H (3×3) that maps points in `frame` to reference frame coordinates.

        Never returns None — falls back to the last known H (or identity) so callers
        can always use the result without None-checks.

        Returns:
            H: np.ndarray of shape (3, 3), dtype float64.
        """
        if self._ref_desc is None:
            raise RuntimeError("Call set_reference() before align()")

        gray = _to_gray(frame)
        kps, desc = self._detect(gray)

        # Strategy 1: direct match to reference (no accumulated drift)
        H = self._estimate_H(self._ref_kps, self._ref_desc, kps, desc)
        if H is not None:
            self.stats["direct"] += 1
            self._update_prev(kps, desc, H)
            return H

        # Strategy 2: incremental — match to previous frame, then compose
        H_to_prev = self._estimate_H(self._prev_kps, self._prev_desc, kps, desc)
        if H_to_prev is not None and self._H_prev_to_ref is not None:
            H = self._H_prev_to_ref @ H_to_prev
            self.stats["incremental"] += 1
            self._update_prev(kps, desc, H)
            return H

        # Fallback: reuse last known H
        self.stats["fallback"] += 1
        return self._H_prev_to_ref if self._H_prev_to_ref is not None else np.eye(3, dtype=np.float64)

    def transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Transform an (N, 2) array of points using homography H.

        Args:
            points: float32 array of shape (N, 2) in current-frame coordinates.
            H:      3×3 homography returned by align().

        Returns:
            Transformed points as float32 array of shape (N, 2).
        """
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, H.astype(np.float32)).reshape(-1, 2)

    def reset(self) -> None:
        """Clear all state. Call between videos."""
        self._ref_kps = []
        self._ref_desc = None
        self._prev_kps = []
        self._prev_desc = None
        self._H_prev_to_ref = None
        self.stats = {"direct": 0, "incremental": 0, "fallback": 0}

    def print_stats(self) -> None:
        total = sum(self.stats.values())
        if total == 0:
            return
        print(
            f"  [FrameAligner] {total} frames aligned — "
            f"direct={self.stats['direct']}, "
            f"incremental={self.stats['incremental']}, "
            f"fallback={self.stats['fallback']}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(self, gray: np.ndarray) -> Tuple[list, Optional[np.ndarray]]:
        kps, desc = self._extractor.detectAndCompute(gray, None)
        return (kps if kps is not None else []), desc

    def _estimate_H(
        self,
        kps_ref: list,
        desc_ref: Optional[np.ndarray],
        kps_cur: list,
        desc_cur: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Estimate H mapping kps_cur coordinates → kps_ref coordinates.
        Returns None if matching or RANSAC fails, or inlier count is too low.
        """
        if desc_ref is None or desc_cur is None:
            return None
        if len(kps_ref) < 4 or len(kps_cur) < 4:
            return None

        try:
            raw = self._matcher.knnMatch(desc_ref, desc_cur, k=2)
        except cv2.error:
            return None

        good = []
        for m_n in raw:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_test * n.distance:
                    good.append(m)

        if len(good) < 4:
            return None

        pts_ref = np.float32([kps_ref[m.queryIdx].pt for m in good])
        pts_cur = np.float32([kps_cur[m.trainIdx].pt for m in good])

        ransac_method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        H, mask = cv2.findHomography(pts_cur, pts_ref, ransac_method, self.ransac_thresh)

        if H is None or mask is None:
            return None
        if int(mask.sum()) < self.min_inliers:
            return None

        return H

    def _update_prev(self, kps: list, desc: Optional[np.ndarray], H: np.ndarray) -> None:
        self._prev_kps = kps
        self._prev_desc = desc
        self._H_prev_to_ref = H


# ------------------------------------------------------------------
# Module-level utility
# ------------------------------------------------------------------

def _to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
