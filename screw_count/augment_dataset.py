import argparse
import re
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# YOLO 标注工具 

def load_yolo_labels(label_path):
    """返回 list of [cls, cx, cy, w, h] (归一化)"""
    labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels


def save_yolo_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for lb in labels:
            f.write(f"{lb[0]} {lb[1]:.6f} {lb[2]:.6f} {lb[3]:.6f} {lb[4]:.6f}\n")


def yolo_to_corners(cx, cy, w, h, W, H):
    """归一化 → 像素角点 (x1,y1,x2,y2)"""
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return x1, y1, x2, y2


def corners_to_yolo(x1, y1, x2, y2, W, H):
    """像素角点 → 归一化 YOLO"""
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return cx, cy, w, h


def clip_labels(labels, W, H, min_size=0.01):
    """裁剪越界 bbox，过滤太小的"""
    valid = []
    for lb in labels:
        cls, cx, cy, w, h = lb
        x1 = max(0, (cx - w / 2) * W)
        y1 = max(0, (cy - h / 2) * H)
        x2 = min(W, (cx + w / 2) * W)
        y2 = min(H, (cy + h / 2) * H)
        nw = (x2 - x1) / W
        nh = (y2 - y1) / H
        if nw > min_size and nh > min_size:
            valid.append([cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, nw, nh])
    return valid


# 增强函数 

def aug_rotate(img, labels, angle=None):
    """任意角度旋转（俯视螺丝旋转不改语义）"""
    if angle is None:
        angle = random.uniform(0, 360)
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(H * sin + W * cos)
    nH = int(H * cos + W * sin)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2

    img_rot = cv2.warpAffine(img, M, (nW, nH),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    new_labels = []
    for lb in labels:
        cls, cx, cy, w, h = lb
        # 旋转 bbox 四角点
        pts = np.array([
            [cx * W - w * W / 2, cy * H - h * H / 2],
            [cx * W + w * W / 2, cy * H - h * H / 2],
            [cx * W + w * W / 2, cy * H + h * H / 2],
            [cx * W - w * W / 2, cy * H + h * H / 2],
        ])
        ones = np.ones((4, 1))
        pts_h = np.hstack([pts, ones])
        rot_pts = (M @ pts_h.T).T
        x1, y1 = rot_pts[:, 0].min(), rot_pts[:, 1].min()
        x2, y2 = rot_pts[:, 0].max(), rot_pts[:, 1].max()
        ncx, ncy, nw, nh = corners_to_yolo(x1, y1, x2, y2, nW, nH)
        new_labels.append([cls, ncx, ncy, nw, nh])

    new_labels = clip_labels(new_labels, nW, nH)
    return img_rot, new_labels


def aug_flip(img, labels, mode='h'):
    """水平/垂直翻转"""
    H, W = img.shape[:2]
    if mode == 'h':
        img_f = cv2.flip(img, 1)
        new_labels = [[lb[0], 1 - lb[1], lb[2], lb[3], lb[4]] for lb in labels]
    else:
        img_f = cv2.flip(img, 0)
        new_labels = [[lb[0], lb[1], 1 - lb[2], lb[3], lb[4]] for lb in labels]
    return img_f, new_labels


def aug_brightness_contrast(img, labels, bright_delta=40, contrast_range=(0.7, 1.3)):
    """亮度+对比度随机扰动（模拟光照变化）"""
    img_f = img.astype(np.float32)
    alpha = random.uniform(*contrast_range)
    beta  = random.uniform(-bright_delta, bright_delta)
    img_f = img_f * alpha + beta
    img_f = np.clip(img_f, 0, 255).astype(np.uint8)
    return img_f, labels


def aug_hsv(img, labels, hue=10, sat=30, val=30):
    """HSV 颜色扰动"""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] + random.randint(-hue, hue), 0, 179)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + random.randint(-sat, sat), 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + random.randint(-val, val), 0, 255)
    img_out = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img_out, labels


def aug_noise(img, labels, sigma=10):
    """高斯噪声"""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img_n = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_n, labels


def aug_blur(img, labels, ksize=None):
    """随机模糊（模拟失焦）"""
    k = ksize or random.choice([3, 5])
    img_b = cv2.GaussianBlur(img, (k, k), 0)
    return img_b, labels


def aug_shadow(img, labels):
    """图片中添加阴影"""
    H, W = img.shape[:2]
    # 随机半边压暗
    x_split = random.randint(W//4, 3*W//4)
    factor = random.uniform(0.4, 0.7)
    side = random.choice(['left', 'right'])
    img = img.copy().astype(np.float32)
    if side == 'left':
        img[:, :x_split] *= factor
    else:
        img[:, x_split:] *= factor
    return np.clip(img, 0, 255).astype(np.uint8), labels


def aug_scale_crop(img, labels, scale_range=(0.7, 1.0)):
    """随机缩放后居中裁回原尺寸"""
    H, W = img.shape[:2]
    scale = random.uniform(*scale_range)
    nW, nH = int(W * scale), int(H * scale)
    img_s = cv2.resize(img, (nW, nH))

    # 填充回原尺寸（白色背景）
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    ox = (W - nW) // 2
    oy = (H - nH) // 2
    canvas[oy:oy + nH, ox:ox + nW] = img_s

    new_labels = []
    for lb in labels:
        cls, cx, cy, w, h = lb
        ncx = (cx * nW + ox) / W
        ncy = (cy * nH + oy) / H
        nw  = w * scale
        nh  = h * scale
        new_labels.append([cls, ncx, ncy, nw, nh])

    new_labels = clip_labels(new_labels, W, H)
    return canvas, new_labels


def aug_cutout(img, labels, n_holes=3, max_size=0.1):
    """随机遮挡块（模拟部分遮挡）"""
    img_c = img.copy()
    H, W = img.shape[:2]
    for _ in range(n_holes):
        sw = int(random.uniform(0.02, max_size) * W)
        sh = int(random.uniform(0.02, max_size) * H)
        x  = random.randint(0, W - sw)
        y  = random.randint(0, H - sh)
        img_c[y:y + sh, x:x + sw] = random.randint(100, 200)
    return img_c, labels


def aug_perspective(img, labels, distort=0.05):
    """轻微透视变换（模拟相机角度偏差）"""
    H, W = img.shape[:2]
    d = distort
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([
        [random.uniform(0, d * W), random.uniform(0, d * H)],
        [W - random.uniform(0, d * W), random.uniform(0, d * H)],
        [W - random.uniform(0, d * W), H - random.uniform(0, d * H)],
        [random.uniform(0, d * W), H - random.uniform(0, d * H)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    img_p = cv2.warpPerspective(img, M, (W, H),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

    new_labels = []
    for lb in labels:
        cls, cx, cy, w, h = lb
        pts = np.array([
            [[cx * W - w * W / 2, cy * H - h * H / 2]],
            [[cx * W + w * W / 2, cy * H - h * H / 2]],
            [[cx * W + w * W / 2, cy * H + h * H / 2]],
            [[cx * W - w * W / 2, cy * H + h * H / 2]],
        ], dtype=np.float32)
        pts_t = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        x1, y1 = pts_t[:, 0].min(), pts_t[:, 1].min()
        x2, y2 = pts_t[:, 0].max(), pts_t[:, 1].max()
        ncx, ncy, nw, nh = corners_to_yolo(x1, y1, x2, y2, W, H)
        new_labels.append([cls, ncx, ncy, nw, nh])

    new_labels = clip_labels(new_labels, W, H)
    return img_p, new_labels


def aug_random_background(img, labels):
    """把白色背景换成随机颜色/纹理，消除背景依赖"""
    H, W = img.shape[:2]
    # 检测接近白色的像素
    white_mask = (img[:,:,0] > 200) & \
                 (img[:,:,1] > 200) & \
                 (img[:,:,2] > 200)
    
    # 随机选择背景类型
    bg_type = random.choice(['color', 'noise', 'gradient'])
    
    if bg_type == 'color':
        # 纯色背景
        color = [random.randint(100,240) for _ in range(3)]
        bg = np.full_like(img, color)
    elif bg_type == 'noise':
        # 噪声背景
        bg = np.random.randint(150, 230, img.shape, dtype=np.uint8)
    else:
        # 渐变背景
        bg = np.zeros_like(img)
        for c in range(3):
            grad = np.linspace(random.randint(150,200),
                               random.randint(200,240), W)
            bg[:,:,c] = np.tile(grad, (H,1)).astype(np.uint8)
    
    result = img.copy()
    result[white_mask] = bg[white_mask]
    return result, labels


def aug_erase_markers(img, labels):
    """用背景色覆盖彩色标记框，避免模型依赖标记框定位"""
    img = img.copy()
    # 检测红色区域 (BGR)
    red_mask = (img[:,:,2] > 150) & \
               (img[:,:,0] < 80) & \
               (img[:,:,1] < 80)
    # 检测蓝色区域
    blue_mask = (img[:,:,0] > 150) & \
                (img[:,:,1] < 100) & \
                (img[:,:,2] < 80)
    # 用白色/浅灰覆盖
    fill = random.randint(200, 240)
    img[red_mask]  = fill
    img[blue_mask] = fill
    return img, labels


# 增强pipeline 

# 每个增强操作: (函数, kwargs, 概率)
AUG_PIPELINE = [
    (aug_rotate,             {},                           1.0),   # 必做，俯视图收益最大
    (aug_flip,               {'mode': 'h'},                0.5),
    (aug_flip,               {'mode': 'v'},                0.5),
    (aug_brightness_contrast,{'bright_delta': 40,           
                              'contrast_range': (0.5,1.5)},0.8),
    (aug_noise,              {'sigma': 15},                0.4),
    (aug_shadow,             {},                           0.5),
    (aug_blur,               {},                           0.3),
    (aug_scale_crop,         {},                           0.5),
    (aug_cutout,             {},                           0.4),
    (aug_perspective,        {'distort': 0.04},            0.2),
    (aug_erase_markers,      {},                           0.3),
]


def apply_pipeline(img, labels):
    """随机应用增强管线中的操作"""
    for fn, kwargs, prob in AUG_PIPELINE:
        if random.random() < prob:
            try:
                img, labels = fn(img, labels, **kwargs)
            except Exception as e:
                print(f"  [warn] {fn.__name__} failed: {e}")
    return img, labels


# 主流程
def build_dataset(img_dir, label_dir, output_dir, aug_per_image, seed=42):
    """
    先划分 train/val，再只对 train 做增强。
    原图全部进 val（干净数据评估），增强图全部进 train。
    """
    random.seed(seed)
    np.random.seed(seed)

    img_dir   = Path(img_dir)
    label_dir = Path(label_dir)
    out       = Path(output_dir)

    # 清理旧数据
    for split in ['train', 'val']:
        split_dir = out / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    print(f"Found {len(img_paths)} original images")
    n_train = 0
    n_val = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] Cannot read {img_path.name}")
            continue

        label_path = label_dir / (img_path.stem + '.txt')
        labels = load_yolo_labels(label_path)
        stem = img_path.stem

        # 原图 → val（干净数据用于验证）
        shutil.copy(img_path, out / 'val' / 'images' / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, out / 'val' / 'labels' / label_path.name)
        n_val += 1

        # 增强图 → train
        for i in range(aug_per_image):
            aug_img, aug_labels = apply_pipeline(img.copy(), [lb[:] for lb in labels])
            out_name = f"{stem}_aug{i:04d}"
            cv2.imwrite(str(out / 'train' / 'images' / f"{out_name}.jpg"), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_yolo_labels(out / 'train' / 'labels' / f"{out_name}.txt", aug_labels)
            n_train += 1

        print(f"  {img_path.name}: val=1(原图), train=+{aug_per_image}(增强)")

    print(f"\nDone! train={n_train}, val={n_val}")
    print(f"  train/images → {out / 'train' / 'images'}")
    print(f"  val/images   → {out / 'val' / 'images'}")
    return n_train, n_val


# CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='螺丝数据集增强工具')
    parser.add_argument('--img_dir',       default='./images',    help='原始图片目录')
    parser.add_argument('--label_dir',     default='./labels',    help='YOLO标注目录')
    parser.add_argument('--output_dir',    default='./augmented', help='输出目录')
    parser.add_argument('--aug_per_image', type=int, default=50,  help='每张图增强几倍')
    parser.add_argument('--seed',          type=int, default=42)
    args = parser.parse_args()

    build_dataset(args.img_dir, args.label_dir, args.output_dir,
                  args.aug_per_image, args.seed)
