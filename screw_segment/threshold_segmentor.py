"""
传统 CV 分割模块
===============
提供多种分割方法：Otsu、自适应阈值、分水岭
"""

import cv2
import numpy as np


class ThresholdSegmentor:
    """螺丝分割器（传统 CV 方法）"""

    def __init__(self, method='otsu', margin=5):
        """
        Parameters
        ----------
        method : str
            分割方法: 'otsu', 'adaptive', 'watershed'
        margin : int
            边界框扩展像素数
        """
        self.method = method
        self.margin = margin
        self._validate_method()

    def segment_with_detections(self, image, detections):
        """
        使用已有检测框进行批量分割（统一接口）。

        Parameters
        ----------
        image : np.ndarray
            原始图像 (BGR)
        detections : list of dict
            来自检测器的输出，至少包含键 'box'

        Returns
        -------
        result : dict
            {
                'full_mask': np.ndarray(uint8, HxW),
                'instance_map': np.ndarray(uint16, HxW),
                'num_instances': int,
                'detections': list,
            }
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        instance_map = np.zeros((h, w), dtype=np.uint16)

        if not detections:
            return {
                'full_mask': full_mask,
                'instance_map': instance_map,
                'num_instances': 0,
                'detections': [],
            }

        inst_id = 1
        for det in detections:
            if 'box' not in det:
                continue

            mask = self.segment_in_bbox(image, det['box'])
            fg = mask > 0
            if fg.sum() == 0:
                continue

            full_mask[fg] = 255
            instance_map[fg] = inst_id
            inst_id += 1

        return {
            'full_mask': full_mask,
            'instance_map': instance_map,
            'num_instances': int(inst_id - 1),
            'detections': detections,
        }

    def _validate_method(self):
        """验证分割方法"""
        valid_methods = ['otsu', 'adaptive', 'watershed']
        if self.method not in valid_methods:
            raise ValueError(f"未知分割方法: {self.method}. 可选: {valid_methods}")

    def segment_in_bbox(self, image, bbox):
        """
        在检测框内进行分割

        Parameters
        ----------
        image : np.ndarray
            原始图像 (BGR)
        bbox : array-like
            边界框 [x1, y1, x2, y2]

        Returns
        -------
        mask : np.ndarray
            分割掩码（与原图同尺寸，仅在 bbox 内有值）
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        # 扩展边界框（带边界检查）
        x1 = max(0, x1 - self.margin)
        y1 = max(0, y1 - self.margin)
        x2 = min(w, x2 + self.margin)
        y2 = min(h, y2 + self.margin)

        # 提取 ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # 转灰度
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 根据方法选择分割策略
        if self.method == 'otsu':
            mask_roi = self._segment_otsu(gray_roi)
        elif self.method == 'adaptive':
            mask_roi = self._segment_adaptive(gray_roi)
        elif self.method == 'watershed':
            mask_roi = self._segment_watershed(roi, gray_roi)

        # 将 ROI 掩码映射回原图
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = mask_roi

        return mask

    def _segment_otsu(self, gray_roi):
        """Otsu 阈值分割"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Otsu 阈值（反向，因为螺丝比背景暗）
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        return binary

    def _segment_adaptive(self, gray_roi):
        """自适应阈值分割"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        return binary

    def _segment_watershed(self, roi, gray_roi):
        """分水岭分割（处理粘连螺丝）"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Otsu 阈值
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 确定前景（种子点）
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 确定背景
        sure_bg = cv2.dilate(binary, kernel, iterations=3)

        # 未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记连通域
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 分水岭
        markers = cv2.watershed(roi, markers)

        # 提取前景掩码（排除边界 -1）
        mask = np.zeros_like(gray_roi)
        mask[markers > 1] = 255

        return mask
