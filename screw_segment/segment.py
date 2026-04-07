"""
混合方法：YOLO 检测 + 传统 CV 分割
==================================
主流程编排模块

用法：
    python segment.py --data_dir ./data --output_dir ./output --weights ../screw_count/weights/best.pt

方法说明：
1. 使用 YOLO 检测模型获取每个螺丝的边界框和类别
2. 在每个检测框内使用传统 CV 方法（阈值 + 分水岭）进行精细分割
3. 输出彩色掩码可视化和实例标签图
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector import ScrewDetector
from threshold_segmentor import ThresholdSegmentor
from sam_segmentor import SamSegmentor
from visualizer import SegmentationVisualizer


class SegmentationPipeline:
    """分割流程管理器"""

    def __init__(
        self,
        detector,
        segmentor,
        visualizer,
        save_instance_map=False
    ):
        """
        Parameters
        ----------
        detector : ScrewDetector
            检测器实例
        segmentor : SamSegmentor/ThresholdSegmentor
            分割器实例
        visualizer : SegmentationVisualizer
            可视化器实例
        save_instance_map : bool
            是否保存实例标签图
        """
        self.detector = detector
        self.segmentor = segmentor
        self.visualizer = visualizer
        self.save_instance_map = save_instance_map

    def process_image(self, img_path):
        """
        处理单张图像

        Parameters
        ----------
        img_path : Path
            图像路径

        Returns
        -------
        result : dict
            包含 'colored_mask', 'instance_map', 'num_instances'
        """
        # 1. 检测螺丝
        image, detections = self.detector.detect(img_path)

        if len(detections) == 0:
            print(f"  未检测到螺丝")
            return {
                'colored_mask': image,
                'instance_map': np.zeros(image.shape[:2], dtype=np.uint16),
                'num_instances': 0
            }

        # 2. 使用统一接口进行批量分割
        seg_result = self.segmentor.segment_with_detections(image, detections)
        instance_map = seg_result['instance_map']

        # 3. 可视化
        colored_mask = self.visualizer.create_colored_mask(image, instance_map, detections)

        return {
            'colored_mask': colored_mask,
            'instance_map': instance_map,
            'num_instances': seg_result['num_instances']
        }

    def process_directory(self, data_dir, output_dir):
        """
        批量处理目录

        Parameters
        ----------
        data_dir : str or Path
            输入图像目录
        output_dir : str or Path
            输出目录
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 查找图像
        img_paths = sorted(data_dir.glob('*.png')) + sorted(data_dir.glob('*.jpg'))
        if not img_paths:
            print(f"未找到图像: {data_dir}")
            return

        print(f"找到 {len(img_paths)} 张图像")
        print(f"检测器: YOLO ({self.detector.weights_path.name})")
        print(f"分割方法: {self.segmentor.method}")
        print("-" * 60)

        total_instances = 0
        start_time = time.time()

        for img_path in img_paths:
            print(f"处理: {img_path.name}")
            t0 = time.time()

            try:
                result = self.process_image(img_path)

                # 保存彩色掩码
                mask_path = output_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), result['colored_mask'])

                # 保存实例标签图（可选）
                if self.save_instance_map:
                    inst_map_path = output_dir / f"{img_path.stem}_instances.png"
                    inst_map_vis = self.visualizer.create_instance_map_visual(result['instance_map'])
                    cv2.imwrite(str(inst_map_path), inst_map_vis)

                    # 保存原始标签数据
                    inst_npy_path = output_dir / f"{img_path.stem}_instances.npy"
                    np.save(str(inst_npy_path), result['instance_map'])

                total_instances += result['num_instances']
                print(f"  检测到 {result['num_instances']} 个实例 | 耗时: {time.time() - t0:.2f}s")

            except Exception as e:
                print(f"  处理失败: {e}")

        total_time = time.time() - start_time

        print("-" * 60)
        print(f"处理完成:")
        print(f"  总图像数: {len(img_paths)}")
        print(f"  总实例数: {total_instances}")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  平均每张: {total_time / len(img_paths):.2f}s")
        print(f"  结果保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='混合方法：YOLO 检测 + 传统 CV 分割')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--segmentor', choices=['threshold', 'sam'], default='sam',
                        help='选择使用的分割算法')
    parser.add_argument('--detector_weights', type=str, default='./weights/best.pt',
                        help='YOLO 检测模型权重路径')
    parser.add_argument('--sam_weights', type=str, default='./weights/sam_vit_l_0b3195.pth', 
                        help='SAM 分割模型权重路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU 阈值')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='YOLO 推理图像尺寸')
    parser.add_argument('--seg_method', type=str, default='otsu',
                        choices=['otsu', 'adaptive', 'watershed'],
                        help='分割方法: otsu(快速), adaptive(适应光照), watershed(处理粘连)')
    parser.add_argument('--margin', type=int, default=5,
                        help='检测框扩展像素数')
    parser.add_argument('--save_instance_map', action='store_true',
                        help='是否保存实例标签图')

    args = parser.parse_args()

    # 初始化各模块
    detector = ScrewDetector(
        weights_path=args.detector_weights,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz
    )

    sam_segmentor = SamSegmentor(
        sam_checkpoint=args.sam_weights,
        model_type='auto',
        device='auto',
        box_margin=2,
        multimask_output=True
    )
    threshold_segmentor = ThresholdSegmentor(
        method=args.seg_method,
        margin=args.margin
    )

    # 选择分割算法
    segmentor = threshold_segmentor
    if args.segmentor == 'sam':
        segmentor = sam_segmentor

    
    visualizer = SegmentationVisualizer(num_classes=5)

    # 创建流程管理器
    pipeline = SegmentationPipeline(
        detector=detector,
        segmentor=segmentor,
        visualizer=visualizer,
        save_instance_map=args.save_instance_map
    )

    # 执行批量处理
    pipeline.process_directory(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
