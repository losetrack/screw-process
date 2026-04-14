"""
VideoProcessor: end-to-end pipeline for a single video.

  1. Open video, pick reference frame (first non-blank frame)
  2. Sample every `frame_step` frames
  3. For each sampled frame: YOLO detect → FrameAligner.align() → GlobalMap.update()
  4. GlobalMap.get_counts() → final per-class counts
  5. Save mask overlay for the middle frame
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow running from project root or screw_video/
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from screw_video.frame_aligner import FrameAligner
from screw_video.global_map import GlobalMap


# Per-class colours for mask overlay (fixed seed, same as screw_segment visualizer)
_CLASS_COLORS = [
    (220,  20,  60),   # Type_1 — crimson
    ( 30, 144, 255),   # Type_2 — dodger blue
    ( 50, 205,  50),   # Type_3 — lime green
    (255, 165,   0),   # Type_4 — orange
    (148,   0, 211),   # Type_5 — violet
]


class VideoProcessor:
    """
    Processes a single video file and returns:
      - counts: List[int] of length 5  (Type_1 … Type_5)
      - mask_frame: np.ndarray BGR image with detection overlay for the middle frame
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        frame_step: int = 5,
        dist_thresh: float = 50.0,
        aligner_method: str = "sift",
    ):
        """
        Args:
            weights_path:   Path to YOLO weights (weights/best.pt).
            conf_thresh:    YOLO confidence threshold.
            iou_thresh:     YOLO NMS IoU threshold.
            imgsz:          YOLO inference image size.
            frame_step:     Process every Nth frame (5 → ~60 frames for a 300-frame video).
            dist_thresh:    GlobalMap dedup distance in reference-frame pixels.
            aligner_method: 'sift' or 'orb' for FrameAligner.
        """
        self._model = _load_yolo(weights_path)
        self._device = _get_device()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.frame_step = frame_step

        self._aligner = FrameAligner(method=aligner_method)
        self._gmap = GlobalMap(dist_thresh=dist_thresh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, video_path: str | Path) -> Tuple[List[int], np.ndarray]:
        """
        Run the full pipeline on one video.

        Returns:
            counts:     [Type_1, Type_2, Type_3, Type_4, Type_5] total counts.
            mask_frame: BGR image — middle frame with detection bounding-box overlay.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_idx = max(total_frames // 2, 0)

        self._aligner.reset()
        self._gmap.reset()

        ref_set = False
        mask_frame: Optional[np.ndarray] = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Reference frame: first sampled frame ---
            if not ref_set and frame_idx % self.frame_step == 0:
                self._aligner.set_reference(frame)
                ref_set = True

            # --- Sampled frames: detect + align + accumulate ---
            if ref_set and frame_idx % self.frame_step == 0:
                detections = self._detect(frame)
                H = self._aligner.align(frame)
                self._gmap.update(detections, H)

                # Save mask overlay for the middle frame (or nearest sampled frame)
                if mask_frame is None and frame_idx >= mid_idx:
                    mask_frame = _draw_mask(frame, detections)

            frame_idx += 1

        cap.release()

        # Fallback: if mid frame was never reached (very short video)
        if mask_frame is None:
            mask_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        counts = self._gmap.get_counts()
        self._aligner.print_stats()
        print(f"  [GlobalMap] raw detections={len(self._gmap)}, "
              f"final counts={counts}")

        return counts, mask_frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> List[dict]:
        """Run YOLO on a BGR frame, return list of {'box', 'class', 'score'}."""
        results = self._model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            verbose=False,
            device=self._device,
        )
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "class": int(box.cls.item()),
                    "score": float(box.conf.item()),
                })
        return detections


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _load_yolo(weights_path: str | Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model = YOLO(str(weights_path))
    print(f"YOLO loaded: {weights_path}")
    return model


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _draw_mask(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draw filled semi-transparent boxes + class labels on frame."""
    vis = frame.copy()
    overlay = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls = det["class"]
        score = det["score"]
        color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)   # filled
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)         # border

        label = f"T{cls + 1} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    return vis
