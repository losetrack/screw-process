"""
可视化模块
=========
负责生成彩色掩码和实例标签图
"""

import cv2
import numpy as np


class SegmentationVisualizer:
    """分割结果可视化器"""

    def __init__(self, num_classes=5):
        """
        Parameters
        ----------
        num_classes : int
            类别数量（用于生成颜色映射）
        """
        self.num_classes = num_classes
        self.colors = self._generate_colors()

    def _generate_colors(self):
        """生成类别颜色映射"""
        np.random.seed(42)
        colors = []
        for _ in range(self.num_classes):
            colors.append(tuple(np.random.randint(50, 255, 3).tolist()))
        return colors

    def create_colored_mask(self, image, instance_map, detections):
        """
        创建彩色掩码可视化

        Parameters
        ----------
        image : np.ndarray
            原始图像
        instance_map : np.ndarray
            实例标签图（每个像素值对应实例 ID）
        detections : list of dict
            检测结果列表

        Returns
        -------
        vis : np.ndarray
            可视化图像（原图 + 半透明掩码 + 边界框）
        """
        vis = image.copy()
        overlay = np.zeros_like(image)

        # 为每个实例绘制掩码
        for inst_id in range(1, instance_map.max() + 1):
            mask = (instance_map == inst_id)
            if mask.sum() == 0:
                continue

            # 获取对应检测的类别
            det_idx = inst_id - 1
            if det_idx < len(detections):
                class_id = detections[det_idx]['class']
                color = self.colors[class_id % len(self.colors)]
            else:
                color = (128, 128, 128)

            # 填充掩码
            overlay[mask] = color

        # 混合原图和掩码
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # 绘制边界框和标签
        for det in detections:
            x1, y1, x2, y2 = det['box'].astype(int)
            class_id = det['class']
            score = det['score']
            color = self.colors[class_id % len(self.colors)]

            # 边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 标签
            label = f"Type_{class_id + 1}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis

    def create_instance_map_visual(self, instance_map):
        """
        创建实例标签图的伪彩色可视化

        Parameters
        ----------
        instance_map : np.ndarray
            实例标签图

        Returns
        -------
        colored : np.ndarray
            伪彩色图像
        """
        # 归一化到 0-255
        if instance_map.max() > 0:
            normalized = (instance_map * 255 // instance_map.max()).astype(np.uint8)
        else:
            normalized = instance_map.astype(np.uint8)

        # 应用伪彩色
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # 背景设为黑色
        colored[instance_map == 0] = 0

        return colored
