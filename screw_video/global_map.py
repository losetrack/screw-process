"""
GlobalMap: accumulates per-frame YOLO detections in reference-frame coordinates,
then deduplicates to produce a final per-class screw count.

Usage:
    gmap = GlobalMap(num_classes=5, dist_thresh=50)
    for frame, H in ...:
        gmap.update(detections, H)
    counts = gmap.get_counts()   # [Type_1, ..., Type_5]
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np


class GlobalMap:
    """
    Maintains a list of detections projected into reference-frame coordinates.
    After all frames are processed, deduplicates nearby detections (same screw
    seen from multiple frames) via greedy NMS on Euclidean distance.

    Each detection is stored as [ref_cx, ref_cy, class_id, confidence].
    """

    def __init__(
        self,
        num_classes: int = 5,
        dist_thresh: float = 50.0,
    ):
        """
        Args:
            num_classes:  Number of screw classes (5 for this project).
            dist_thresh:  Two detections closer than this (pixels, in reference
                          frame coords) are considered the same screw instance.
                          Tune on dev videos — ~50px works for typical top-down
                          footage where screws are well-separated.
        """
        self.num_classes = num_classes
        self.dist_thresh = dist_thresh

        # Raw accumulated detections: list of [cx, cy, class_id, confidence]
        self._detections: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[dict],
        H: np.ndarray,
    ) -> None:
        """
        Project this frame's detections into reference coordinates and accumulate.

        Args:
            detections: list of dicts from ScrewDetector, each with keys:
                          'box'   — [x1, y1, x2, y2] in current-frame pixels
                          'class' — int class index (0-based)
                          'score' — float confidence
            H:          3×3 homography from FrameAligner.align(), mapping
                        current-frame coords → reference-frame coords.
        """
        if not detections:
            return

        # Collect box centres in current-frame coords
        centres = np.array(
            [[(d["box"][0] + d["box"][2]) / 2,
              (d["box"][1] + d["box"][3]) / 2]
             for d in detections],
            dtype=np.float32,
        )  # (N, 2)

        # Project to reference frame
        ref_centres = _transform_points(centres, H)  # (N, 2)

        for i, d in enumerate(detections):
            cls = int(d["class"])
            if 0 <= cls < self.num_classes:
                self._detections.append(
                    np.array([ref_centres[i, 0], ref_centres[i, 1], cls, float(d["score"])],
                             dtype=np.float64)
                )

    def get_counts(self) -> List[int]:
        """
        Deduplicate accumulated detections and return per-class counts.

        Deduplication strategy — greedy distance-based NMS:
          1. Sort all detections by confidence (descending).
          2. Greedily keep each detection if no already-kept detection of the
             same class is within dist_thresh pixels.
          3. Count kept detections per class.

        Returns:
            List of length num_classes: [count_class_0, ..., count_class_N-1]
        """
        if not self._detections:
            return [0] * self.num_classes

        dets = np.stack(self._detections)  # (M, 4): cx, cy, cls, score
        # Sort by confidence descending
        order = np.argsort(-dets[:, 3])
        dets = dets[order]

        kept = []  # list of (cx, cy, cls)
        for det in dets:
            cx, cy, cls = det[0], det[1], int(det[2])
            if not _is_suppressed(cx, cy, cls, kept, self.dist_thresh):
                kept.append((cx, cy, cls))

        counts = [0] * self.num_classes
        for _, _, cls in kept:
            counts[cls] += 1
        return counts

    def reset(self) -> None:
        """Clear all accumulated detections. Call between videos."""
        self._detections.clear()

    def __len__(self) -> int:
        return len(self._detections)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform (N, 2) float32 points with homography H. Returns (N, 2)."""
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H.astype(np.float32)).reshape(-1, 2)


def _is_suppressed(
    cx: float,
    cy: float,
    cls: int,
    kept: list,
    thresh: float,
) -> bool:
    """Return True if (cx, cy, cls) is within thresh pixels of any kept detection of the same class."""
    for kx, ky, kcls in kept:
        if kcls == cls:
            if (cx - kx) ** 2 + (cy - ky) ** 2 < thresh ** 2:
                return True
    return False


