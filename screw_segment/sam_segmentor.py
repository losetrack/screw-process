"""
SAM 分割模块
=============
基于检测框进行螺丝实例分割：
1. 调用检测器获取螺丝边界框
2. 将检测框输入 SAM，得到每个螺丝实例 mask
3. 聚合为整图二值掩码和实例标签图
"""

from pathlib import Path

import cv2
import numpy as np


class SamSegmentor:
	"""螺丝分割器（SAM + 检测框提示）"""

	def __init__(
		self,
		sam_checkpoint,
		model_type="auto",
		device="auto",
		box_margin=2,
		multimask_output=True,
	):
		"""
		Parameters
		----------
		sam_checkpoint : str or Path
			SAM 权重路径
		model_type : str
			SAM 模型类型: 'auto'|'vit_h'|'vit_l'|'vit_b'
		device : str
			推理设备: 'auto'|'cuda'|'cpu'
		box_margin : int
			检测框扩展像素数
		multimask_output : bool
			是否启用多候选 mask，并选择得分最高者
		"""
		self.sam_checkpoint = Path(sam_checkpoint)
		self.model_type = model_type
		self.device = device
		self.box_margin = int(box_margin)
		self.multimask_output = bool(multimask_output)

		self._torch, self._predictor = self._build_predictor()

	@staticmethod
	def _infer_model_type_from_checkpoint(checkpoint_path):
		name = checkpoint_path.name.lower()
		if "vit_h" in name:
			return "vit_h"
		if "vit_l" in name:
			return "vit_l"
		if "vit_b" in name:
			return "vit_b"
		return None

	def _build_predictor(self):
		try:
			import torch
			from segment_anything import SamPredictor, sam_model_registry
		except ImportError as exc:
			raise ImportError(
				"请安装依赖: pip install torch torchvision segment-anything opencv-python"
			) from exc

		if not self.sam_checkpoint.exists():
			raise FileNotFoundError(f"SAM 权重不存在: {self.sam_checkpoint}")

		if self.model_type == "auto":
			inferred = self._infer_model_type_from_checkpoint(self.sam_checkpoint)
			if inferred is None:
				raise ValueError(
					"无法从权重文件名推断 model_type，请显式传入 vit_h|vit_l|vit_b"
				)
			model_type = inferred
		else:
			model_type = self.model_type

		if self.device == "auto":
			device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			device = self.device

		try:
			sam = sam_model_registry[model_type](checkpoint=str(self.sam_checkpoint))
		except RuntimeError as exc:
			raise RuntimeError(
				"SAM 权重与 model_type 不匹配。\n"
				f"当前权重: {self.sam_checkpoint.name}\n"
				f"当前 model_type: {model_type}\n"
				"示例: sam_vit_l_*.pth -> vit_l"
			) from exc

		sam.to(device=device)
		predictor = SamPredictor(sam)
		self.model_type = model_type
		self.device = device
		return torch, predictor

	def _expand_bbox(self, bbox, image_shape):
		h, w = image_shape[:2]
		x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).tolist()

		x1 = max(0, int(np.floor(x1 - self.box_margin)))
		y1 = max(0, int(np.floor(y1 - self.box_margin)))
		x2 = min(w - 1, int(np.ceil(x2 + self.box_margin)))
		y2 = min(h - 1, int(np.ceil(y2 + self.box_margin)))

		if x2 <= x1 or y2 <= y1:
			return None
		return np.array([x1, y1, x2, y2], dtype=np.float32)

	def segment_in_bbox(self, image_bgr, bbox):
		"""
		在单个检测框内分割并返回整图大小掩码。

		Parameters
		----------
		image_bgr : np.ndarray
			原始图像 (BGR)
		bbox : array-like
			检测框 [x1, y1, x2, y2]

		Returns
		-------
		mask : np.ndarray
			uint8 掩码（与原图同尺寸，前景为 255）
		"""
		if image_bgr is None or image_bgr.size == 0:
			raise ValueError("输入图像为空")

		full_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
		box = self._expand_bbox(bbox, image_bgr.shape)
		if box is None:
			return full_mask

		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		self._predictor.set_image(image_rgb)

		masks, scores, _ = self._predictor.predict(
			box=box,
			multimask_output=self.multimask_output,
		)

		best_idx = int(np.argmax(scores))
		full_mask[masks[best_idx]] = 255
		return full_mask

	def segment_with_detections(self, image_bgr, detections):
		"""
		使用已有检测框进行批量分割。

		Parameters
		----------
		image_bgr : np.ndarray
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
		if image_bgr is None or image_bgr.size == 0:
			raise ValueError("输入图像为空")

		h, w = image_bgr.shape[:2]
		full_mask = np.zeros((h, w), dtype=np.uint8)
		instance_map = np.zeros((h, w), dtype=np.uint16)

		if not detections:
			return {
				"full_mask": full_mask,
				"instance_map": instance_map,
				"num_instances": 0,
				"detections": [],
			}

		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		self._predictor.set_image(image_rgb)

		inst_id = 1
		for det in detections:
			if "box" not in det:
				continue

			box = self._expand_bbox(det["box"], image_bgr.shape)
			if box is None:
				continue

			masks, scores, _ = self._predictor.predict(
				box=box,
				multimask_output=self.multimask_output,
			)
			best_idx = int(np.argmax(scores))
			best_mask = masks[best_idx]

			full_mask[best_mask] = 255
			instance_map[best_mask] = inst_id
			inst_id += 1

		return {
			"full_mask": full_mask,
			"instance_map": instance_map,
			"num_instances": int(inst_id - 1),
			"detections": detections,
		}

	def detect_and_segment(self, img_path, detector):
		"""
		端到端流程：检测 + SAM 分割。

		Parameters
		----------
		img_path : str or Path
			输入图像路径
		detector : object
			检测器实例，需实现 detect(img_path) -> (image, detections)

		Returns
		-------
		result : dict
			在 segment_with_detections 返回结果基础上，额外包含 'image'
		"""
		image_bgr, detections = detector.detect(img_path)
		result = self.segment_with_detections(image_bgr, detections)
		result["image"] = image_bgr
		return result
