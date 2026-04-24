"""
YOLO-based screw detector
"""
from ultralytics import YOLO
import numpy as np
import torch


class ScrewDetector:
    """Wrapper for YOLO screw detection model"""

    def __init__(self, weights_path, conf=0.25, iou=0.45, imgsz=640, device=None):
        """
        Args:
            weights_path: Path to YOLO weights file
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
        """
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def detect(self, frame):
        """
        Detect screws in a single frame

        Args:
            frame: Input image (BGR format)

        Returns:
            List of detections, each as {'box': [x1,y1,x2,y2], 'class': int, 'score': float}
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]

        detections = []
        if results.boxes is not None:
            # Move all prediction fields to CPU in one shot to avoid per-box sync overhead.
            xyxy = results.boxes.xyxy.detach().cpu().numpy()
            classes = results.boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
            scores = results.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)

            for box, class_id, score in zip(xyxy, classes, scores):
                detections.append({
                    'box': box,
                    'class': int(class_id),
                    'score': float(score)
                })

        return detections
