"""
基于 SAM 的交互式实例分割脚本

功能：
1. 鼠标左键点击图像，生成点击区域的实例 mask。
2. 退出交互时，自动对整张图像生成 SAM 全图实例 mask。

示例：
	python sam_seg.py --image ./data/image_1.png --sam_checkpoint ./weights/sam_vit_b_01ec64.pth
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_argparser():
	parser = argparse.ArgumentParser(description="SAM 交互式实例分割")
	parser.add_argument("--image", type=str, required=True, help="输入图像路径")
	parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
	parser.add_argument(
		"--sam_checkpoint",
		type=str,
		required=True,
		help="SAM 权重路径，例如 sam_vit_b_01ec64.pth",
	)
	parser.add_argument(
		"--model_type",
		type=str,
		default="auto",
		choices=["auto", "vit_h", "vit_l", "vit_b"],
		help="SAM 模型类型，默认 auto 会从权重文件名推断",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		choices=["auto", "cuda", "cpu"],
		help="推理设备",
	)
	parser.add_argument(
		"--window_name",
		type=str,
		default="SAM Click Segmentation",
		help="交互窗口名称",
	)
	return parser


def infer_model_type_from_checkpoint(checkpoint_path):
	name = checkpoint_path.name.lower()
	if "vit_h" in name:
		return "vit_h"
	if "vit_l" in name:
		return "vit_l"
	if "vit_b" in name:
		return "vit_b"
	return None


def random_color(seed):
	rng = np.random.default_rng(seed)
	return tuple(int(x) for x in rng.integers(40, 255, size=3))


def overlay_instances(base_bgr, masks):
	vis = base_bgr.copy()
	alpha = 0.45
	for idx, mask in enumerate(masks, start=1):
		color = random_color(idx)
		color_layer = np.zeros_like(vis, dtype=np.uint8)
		color_layer[mask] = color
		vis = cv2.addWeighted(vis, 1.0, color_layer, alpha, 0)
	return vis


def save_click_results(image_stem, output_dir, image_bgr, click_masks, click_points):
	h, w = image_bgr.shape[:2]
	instance_map = np.zeros((h, w), dtype=np.uint16)

	for inst_id, mask in enumerate(click_masks, start=1):
		instance_map[mask] = inst_id

	click_vis = overlay_instances(image_bgr, click_masks)
	for x, y in click_points:
		cv2.circle(click_vis, (x, y), 4, (0, 255, 255), -1)

	cv2.imwrite(str(output_dir / f"{image_stem}_click_mask.png"), click_vis)
	np.save(str(output_dir / f"{image_stem}_click_instances.npy"), instance_map)


def save_auto_results(image_stem, output_dir, image_bgr, auto_masks):
	h, w = image_bgr.shape[:2]
	instance_map = np.zeros((h, w), dtype=np.uint16)

	sorted_masks = sorted(auto_masks, key=lambda m: m["area"], reverse=True)
	for inst_id, ann in enumerate(sorted_masks, start=1):
		instance_map[ann["segmentation"]] = inst_id

	bool_masks = [ann["segmentation"] for ann in sorted_masks]
	auto_vis = overlay_instances(image_bgr, bool_masks)
	cv2.imwrite(str(output_dir / f"{image_stem}_auto_mask.png"), auto_vis)
	np.save(str(output_dir / f"{image_stem}_auto_instances.npy"), instance_map)


def main():
	args = build_argparser().parse_args()

	try:
		import torch
		from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
	except ImportError as exc:
		raise ImportError(
			"缺少依赖。请先安装: pip install torch torchvision opencv-python segment-anything"
		) from exc

	image_path = Path(args.image)
	if not image_path.exists():
		raise FileNotFoundError(f"找不到输入图像: {image_path}")

	checkpoint_path = Path(args.sam_checkpoint)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"找不到 SAM 权重: {checkpoint_path}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	image_bgr = cv2.imread(str(image_path))
	if image_bgr is None:
		raise RuntimeError(f"图像读取失败: {image_path}")
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

	if args.device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device

	if args.model_type == "auto":
		inferred = infer_model_type_from_checkpoint(checkpoint_path)
		if inferred is None:
			raise ValueError(
				"无法从权重文件名推断模型类型，请显式指定 --model_type vit_h|vit_l|vit_b"
			)
		model_type = inferred
	else:
		model_type = args.model_type

	print(f"SAM model_type: {model_type} | device: {device}")

	try:
		sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
	except RuntimeError as exc:
		raise RuntimeError(
			"SAM 权重与 model_type 不匹配。\n"
			f"当前权重: {checkpoint_path.name}\n"
			f"当前 model_type: {model_type}\n"
			"请检查并使用匹配组合，例如:\n"
			"  sam_vit_l_*.pth -> --model_type vit_l\n"
			"  sam_vit_b_*.pth -> --model_type vit_b\n"
			"  sam_vit_h_*.pth -> --model_type vit_h"
		) from exc

	sam.to(device=device)

	predictor = SamPredictor(sam)
	predictor.set_image(image_rgb)

	click_masks = []
	click_points = []
	latest_vis = image_bgr.copy()

	def refresh_display():
		nonlocal latest_vis
		latest_vis = overlay_instances(image_bgr, click_masks)
		for x, y in click_points:
			cv2.circle(latest_vis, (x, y), 4, (0, 255, 255), -1)
		cv2.putText(
			latest_vis,
			"LClick: add mask | U: undo | C: clear | Q/ESC: quit & auto mask",
			(10, 28),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.65,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)
		cv2.imshow(args.window_name, latest_vis)

	def on_mouse(event, x, y, _flags, _param):
		if event != cv2.EVENT_LBUTTONDOWN:
			return

		point_coords = np.array([[x, y]])
		point_labels = np.array([1])

		masks, scores, _ = predictor.predict(
			point_coords=point_coords,
			point_labels=point_labels,
			multimask_output=True,
		)
		best_idx = int(np.argmax(scores))
		best_mask = masks[best_idx]

		click_masks.append(best_mask)
		click_points.append((x, y))
		refresh_display()

	cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
	cv2.setMouseCallback(args.window_name, on_mouse)
	refresh_display()

	while True:
		key = cv2.waitKey(20) & 0xFF

		if key in (ord("q"), 27):
			break
		if key == ord("u") and click_masks:
			click_masks.pop()
			click_points.pop()
			refresh_display()
		if key == ord("c"):
			click_masks.clear()
			click_points.clear()
			refresh_display()

	cv2.destroyAllWindows()

	print("正在保存点击结果...")
	save_click_results(image_path.stem, output_dir, image_bgr, click_masks, click_points)

	print("正在生成整图自动 mask（SAM AutomaticMaskGenerator）...")
	mask_generator = SamAutomaticMaskGenerator(sam)
	auto_masks = mask_generator.generate(image_rgb)
	save_auto_results(image_path.stem, output_dir, image_bgr, auto_masks)

	print("完成。输出文件:")
	print(f"  - {output_dir / (image_path.stem + '_click_mask.png')}")
	print(f"  - {output_dir / (image_path.stem + '_click_instances.npy')}")
	print(f"  - {output_dir / (image_path.stem + '_auto_mask.png')}")
	print(f"  - {output_dir / (image_path.stem + '_auto_instances.npy')}")


if __name__ == "__main__":
	main()
