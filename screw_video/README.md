# Screw Video Counting

基于 `YOLO + ByteTrack + OSNet Re-ID` 的视频螺丝分类计数项目，用于对多段视频中的 5 类螺丝进行检测、跟踪、去重计数，并输出结果文件、处理时间和可视化掩膜图。

## 1. 项目目标

输入一个包含测试视频的文件夹，程序会逐段视频完成：

- 螺丝检测
- 多目标跟踪
- 跨轨迹 Re-ID 匹配
- 按类别去重计数
- 输出中间帧掩膜图

最终输出：

- `result.npy`：每段视频的 5 类螺丝计数结果
- `time.txt`：处理所有视频的总耗时
- `mask_folder/*.png`：每段视频对应的一张掩膜可视化图

## 2. 整体流程

当前实现逻辑如下：

1. 使用 YOLO 检测每帧中的螺丝目标，输出检测框、类别和置信度。
2. 使用 ByteTrack 对检测结果做帧间关联，形成局部轨迹。
3. 对轨迹引入 OSNet Re-ID 特征，用于跨断轨匹配和全局 ID 维护。
4. 为降低 Re-ID 开销，当前策略为：
   - 新轨迹首次出现时提取特征
   - 已存在轨迹每隔 `N` 帧更新一次特征，默认 `5` 帧
5. 对位于画面边缘的检测框可做过滤，减少因截断导致的误检测和误关联。
6. 对轨迹类别做历史投票，得到最终类别。
7. 对过短轨迹做过滤，避免短暂误检造成重复计数。
8. 最终按 `[Type_1, Type_2, Type_3, Type_4, Type_5]` 输出每段视频的计数。

## 3. 目录结构

```text
screw_video/
  run.py                         # 主入口，批量处理视频并输出结果
  test.py                        # 可视化调试脚本
  detector.py                    # YOLO 检测封装
  tracker.py                     # ByteTrack + Re-ID 跟踪封装
  reid.py                        # OSNet Re-ID 特征提取与全局 ID 匹配
  counter.py                     # 计数逻辑
  visualizer.py                  # 结果绘制
  requirements.txt               # 依赖
  README.md                      # 使用说明
  weights/                       # 检测与 Re-ID 权重
  vedio_exp/                     # 输入视频目录
  mask_folder/                   # 输出掩膜图目录
  vis_videos/                    # 调试视频输出目录
  reid_dataset/                  # 原始 Re-ID crop 数据
  reid_dataset_torchreid/        # torchreid 训练格式数据
  reid_runs/                     # Re-ID 训练输出
  script/
    export_reid_crops.py         # 从当前流程自动导出 Re-ID crop
    prepare_torchreid_dataset.py # 转换为 torchreid/Market1501 风格
    train_torchreid.py           # OSNet Re-ID 训练脚本
```

## 4. 环境依赖

推荐 Python 3.10 及以上。

安装依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 包含：

- `ultralytics`
- `boxmot`
- `torchreid`
- `opencv-python`
- `numpy`
- `torch`

建议准备好可用的 CUDA 环境。当前代码中：

- YOLO 检测默认优先使用 `cuda:0`
- Re-ID 默认优先使用 CUDA
- 若无可用 GPU，会自动退回 CPU

## 5. 快速开始

### 5.1 主流程运行

按作业要求运行：

```bash
python run.py --data_dir ./vedio_exp --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder
```

如果希望显式指定检测与 Re-ID 都用 GPU：

```bash
python run.py --data_dir ./vedio_exp --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder --detector_device cuda:0 --reid_device cuda:0
```

如果启用训练后的 Re-ID 权重：

```bash
python run.py --data_dir ./vedio_exp --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder --reid_weights weights/osnet_x0_25_screw_reid.pth.tar
```

### 5.2 可视化调试

```bash
python test.py --data_dir ./vedio_exp --output_dir ./vis_videos --reid_weights weights/osnet_x0_25_screw_reid.pth.tar
```

常用调试参数：

- `--max_frames`：只跑前若干帧
- `--display`：实时显示
- `--no_save_video`：不写调试视频，加快测试
- `--sample_interval`：按间隔抽帧处理

示例：

```bash
python test.py --data_dir ./vedio_exp --output_dir ./vis_videos --max_frames 300 --no_save_video --display --reid_weights weights/osnet_x0_25_screw_reid.pth.tar
```

## 6. 输出说明

### 6.1 `result.npy`

加载方式：

```python
import numpy as np
result = np.load("result.npy", allow_pickle=True).item()
print(result)
```

结果格式示例：

```python
{
    "IMG_2374": [14, 7, 6, 22, 3],
    "IMG_2375": [15, 9, 9, 17, 4],
    "IMG_2376": [15, 5, 6, 12, 4],
}
```

其中 value 按顺序对应：

```text
[Type_1, Type_2, Type_3, Type_4, Type_5]
```

### 6.2 `time.txt`

记录全部视频处理总时间，单位为秒。

### 6.3 `mask_folder`

每段视频会输出一张命名为 `{video_name}_mask.png` 的可视化图，用于定性展示检测与跟踪效果。

## 7. 主要参数说明

### 7.1 检测相关

- `--weights`：YOLO 权重路径
- `--conf`：检测置信度阈值
- `--iou`：NMS 阈值
- `--imgsz`：YOLO 输入尺寸
- `--detector_device`：检测设备，如 `cuda:0` 或 `cpu`

### 7.2 跟踪相关

- `--track_thresh`：ByteTrack 检测阈值
- `--track_buffer`：丢失轨迹保留帧数
- `--match_thresh`：ByteTrack IOU 匹配阈值
- `--sample_interval`：每隔多少帧处理一次
- `--edge_margin`：忽略边缘检测框的像素边界
- `--min_track_length`：计数时忽略过短轨迹

### 7.3 Re-ID 相关

- `--reid_weights`：Re-ID 权重路径
- `--reid_model`：OSNet 模型名，默认 `osnet_x0_25`
- `--reid_match_thresh`：Re-ID 相似度阈值
- `--reid_max_age`：跨轨迹重连允许的最大时间间隔
- `--reid_update_interval`：已有轨迹更新特征的帧间隔
- `--reid_device`：Re-ID 设备，如 `cuda:0` 或 `cpu`

## 8. Re-ID 数据准备与训练

### 8.1 导出 Re-ID crop

从现有检测跟踪流程中自动导出 crop：

```bash
python script/export_reid_crops.py --data_dir ./vedio_exp --output_dir ./reid_dataset --min_track_length 5
```

输出格式：

```text
reid_dataset/
  train/
    IMG_2374_tid0001/
    IMG_2374_tid0002/
    ...
  metadata.csv
```

### 8.2 转换为 torchreid 训练格式

```bash
python script/prepare_torchreid_dataset.py --input_dir ./reid_dataset/train --output_dir ./reid_dataset_torchreid
```

输出为 Market1501 风格目录：

```text
reid_dataset_torchreid/
  market1501/
    Market-1501-v15.09.15/
      bounding_box_train/
      query/
      bounding_box_test/
  split_summary.csv
```

### 8.3 训练 Re-ID 模型

```bash
python script/train_torchreid.py
```

若先用 CPU 或小批量测试：

```bash
python script/train_torchreid.py --use_cpu --workers 0 --batch_size_train 4 --batch_size_test 4
```

训练完成后，默认导出权重到：

```text
weights/osnet_x0_25_screw_reid.pth.tar
```

然后可直接用于主流程：

```bash
python run.py --data_dir ./vedio_exp --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder --reid_weights weights/osnet_x0_25_screw_reid.pth.tar
```

## 9. 当前实现中的速度优化

当前代码已经包含以下速度相关优化：

- 检测模块显式优先使用 GPU
- 检测后处理改为批量 `.cpu().numpy()` 提取，减少逐框同步
- Re-ID 改为新轨迹首次提特征，老轨迹按间隔更新
- 支持 `sample_interval` 抽帧处理
- 可过滤边缘框，减少无效跟踪与误匹配

如果需要进一步提速，优先建议：

- 增大 `sample_interval`
- 增大 `reid_update_interval`
- 关闭 `test.py` 中的视频写出和显示
- 适当减小 `imgsz`

## 10. 已知说明

- `torchreid` 在评估时可能提示 `Cython evaluation is unavailable`，这是性能提示，不影响训练和推理正确性。
- 加载 ImageNet 预训练 OSNet 权重时，`classifier.weight` 和 `classifier.bias` 被丢弃是正常现象，因为训练类别数与预训练分类头不一致。
- `test.py` 是调试脚本，包含绘制、写视频、显示窗口，运行速度会明显慢于 `run.py`。

## 11. 团队信息

请在提交前补充：

```text
成员1：姓名 + 学号
成员2：姓名 + 学号
成员3：姓名 + 学号
成员4：姓名 + 学号
```

