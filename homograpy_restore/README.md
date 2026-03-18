# 作业1：单应性恢复与多余螺丝去除

本目录包含用于将透视变形图像恢复到模板视角的脚本，并支持自动去除恢复结果中相对于模板多出来的螺丝及其阴影区域。

## 环境要求

- Python 3.8 及以上
- OpenCV（cv2）
- NumPy

## 基本用法

```bash
python ./restore_without_extra_screws.py --template data/template.png --input_dir data/ --output_dir restored_images/
```

脚本默认会处理 input_dir 中匹配 raw_*_warp*.png 的图像，并将结果写入 output_dir。

## 参数说明

- --template：模板图像路径（必填）
- --input_dir：输入图像目录（必填）
- --output_dir：输出目录（必填）
- --pattern：输入文件匹配模式，默认 raw_*_warp*.png
- --debug：开启调试输出（可选）
- --remove-screws: 开启是否消除多余螺丝（可选）

## 核心流程

1. 读取模板图像并生成模板螺丝掩码。
2. 对每张输入图像提取特征并进行匹配。
3. 使用比值测试与 RANSAC 估计单应矩阵，将输入图像透视变换到模板坐标系。
4. 检测恢复图像中的螺丝区域，和模板螺丝区域进行对比，定位多余螺丝。
5. 对多余螺丝区域进行膨胀并结合距离变换做平滑融合，用模板内容替换该区域。
6. 保存最终恢复结果。
