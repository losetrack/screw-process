"""
YOLO-based screw detector
"""
from ultralytics import YOLO
import numpy as np


class ScrewDetector:
    """Wrapper for YOLO screw detection model"""

    def __init__(self, weights_path, conf=0.25, iou=0.45, imgsz=640):
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
            verbose=False
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                detections.append({
                    'box': box.xyxy[0].cpu().numpy(),
                    'class': int(box.cls),
                    'score': float(box.conf)
                })

        return detections
