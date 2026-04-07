"""
YOLO 检测模块
============
负责加载 YOLO 模型并执行螺丝检测
"""

import cv2
import numpy as np
from pathlib import Path


class ScrewDetector:
    """螺丝检测器（基于 YOLO）"""

    def __init__(self, weights_path, conf_thresh=0.25, iou_thresh=0.45, imgsz=640):
        """
        Parameters
        ----------
        weights_path : str or Path
            YOLO 模型权重路径
        conf_thresh : float
            置信度阈值
        iou_thresh : float
            NMS IoU 阈值
        imgsz : int
            推理图像尺寸
        """
        self.weights_path = Path(weights_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.model = self._load_model()

    def _load_model(self):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装: pip install ultralytics")

        if not self.weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {self.weights_path}")

        model = YOLO(str(self.weights_path))
        print(f"YOLO 模型已加载: {self.weights_path}")
        return model

    def _get_device(self):
        """获取推理设备"""
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'

    def detect(self, img_path):
        """
        检测图像中的螺丝

        Parameters
        ----------
        img_path : str or Path
            图像路径

        Returns
        -------
        image : np.ndarray
            原始图像 (BGR)
        detections : list of dict
            每个检测结果包含: {'box': [x1,y1,x2,y2], 'class': int, 'score': float}
        """
        results = self.model.predict(
            source=str(img_path),
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            verbose=False,
            device=self._get_device(),
        )

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    'box': box.xyxy[0].cpu().numpy(),
                    'class': int(box.cls.item()),
                    'score': float(box.conf.item())
                })

        return image, detections
