
**数据集描述：
军事坦克检测数据集
一共2221张数据集，yolo格式标注。**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/994ffae49f404f3686a0cb23c5753dad.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/85061ecbfce64fcbb92c79cae59fc30c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2621f14d4c7a44bcb97901835379210b.png)
对该数据集的 **完整训练流程、详细代码（基于 Ultralytics YOLOv8）**，包括：环境配置、数据组织、模型训练、验证、推理和部署。代码示例，仅供参考学习使用。

---

## ✅ 一、数据集结构与 `data.yaml` 配置

### 1. 博主推荐目录结构

```
tank_dataset/
├── images/
│   ├── train/      # 约 1777 张 (80%)
│   └── val/        # 约 444 张 (20%)
├── labels/
│   ├── train/
│   └── val/
└── data.yaml       # 数据集配置文件
```

> 📌 建议按 **8:2 划分训练集和验证集**

---

### 2. 创建 `data.yaml`

```yaml
# data.yaml
train: ./tank_dataset/images/train
val: ./tank_dataset/images/val

# 类别数量
nc: 1

# 类别名称
names:
  - Tank
```

> ⚠️ 注意：
> - 所有 `.txt` 文件必须与图像同名。
> - 每行格式为：`0 x_center y_center w h`（归一化到 [0,1]）
> - 如果你的标签中已有类别 ID，请确保是 `0`（因为只有一类）

---

## ✅ 二、环境准备（推荐 Anaconda）

```bash
# 创建虚拟环境
conda create -n tank_det python=3.9
conda activate tank_det

# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 YOLOv8
pip install ultralytics opencv-python matplotlib tqdm

# 验证CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

---

## ✅ 三、自动划分数据集脚本（如果尚未划分）

如果你的数据在同一个文件夹中，运行以下脚本自动划分：

```python
# split_tank_dataset.py
import os
import random
import shutil
from pathlib import Path

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # 支持 jpg, png, jpeg
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    # 创建目录
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    # 复制图像和标签
    for img in train_images:
        shutil.copy(img, output_dir / 'images' / 'train' / img.name)
        label = label_dir / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, output_dir / 'labels' / 'train' / label.name)

    for img in val_images:
        shutil.copy(img, output_dir / 'images' / 'val' / img.name)
        label = label_dir / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, output_dir / 'labels' / 'val' / label.name)

    print(f"✅ 数据集划分完成！")
    print(f"   训练集: {len(train_images)} 张")
    print(f"   验证集: {len(val_images)} 张")

# 使用示例
split_dataset(
    image_dir='raw/images',
    label_dir='raw/labels',
    output_dir='tank_dataset'
)
```

---

## ✅ 四、完整训练代码（YOLOv8）

```python
from ultralytics import YOLO
import torch

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载预训练模型（推荐 yolov8m 或 yolov8l 提升小目标检测）
model = YOLO('yolov8m.pt')  # 可选: yolov8s.pt, yolov8l.pt, yolov8x.pt

# 开始训练
results = model.train(
    data='tank_dataset/data.yaml',     # 数据配置文件
    imgsz=640,                         # 输入尺寸（若坦克较小可尝试 800 或 1024）
    epochs=150,                        # 训练轮数
    batch=16,                          # 批次大小（根据显存调整）
    name='yolov8m_tank_detection',
    project='runs/tank',
    device=device,
    workers=8,                         # 数据加载线程
    optimizer='AdamW',                 # 更稳定优化器
    lr0=0.001,                         # 初始学习率
    momentum=0.937,
    weight_decay=0.0005,
    patience=30,                       # 早停：30轮无提升则停止
    save_period=10,                    # 每10轮保存一次
    exist_ok=True,
    augment=True,                      # 启用Mosaic/HSV增强
    close_mosaic=20,                   # 最后20轮关闭Mosaic
    amp=True,                          # 混合精度训练
    val=True                           # 每轮验证
)

print("✅ 坦克检测模型训练完成！")
```

---

## ✅ 五、关键参数说明（适用于小目标/军事目标）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `imgsz` | `640` 或 `800` | 若坦克较小建议 `800` |
| `batch` | `8~32` | 显存越大越好 |
| `epochs` | `100~200` | 小数据集建议多训 |
| `optimizer` | `AdamW` | 比SGD更稳定 |
| `lr0` | `0.001` | 初始学习率 |
| `augment` | `True` | Mosaic增强对小目标有效 |
| `close_mosaic` | `20` | 后期关闭提升稳定性 |

---

## ✅ 六、模型验证（评估性能）

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('runs/tank/yolov8m_tank_detection/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data='tank_dataset/data.yaml',
    batch=16,
    imgsz=640,
    device='cuda',
    project='val_results',
    name='tank_val',
    save_json=True
)

# 输出关键指标
print(f"mAP@0.5:       {metrics.box.map:.4f}")
print(f"mAP@0.5:0.95:  {metrics.box.map50_95:.4f}")
print(f"Precision:     {metrics.box.precision:.4f}")
print(f"Recall:        {metrics.box.recall:.4f}")
print(f"F1 Score:      {metrics.box.f1.mean():.4f}")
print(f"FPS:           {metrics.speed['inference']:.2f}")
```

---

## ✅ 七、推理代码（图像、视频、摄像头）

### 1. 单张图像推理

```python
results = model.predict(
    source='test_tank.jpg',
    conf=0.3,
    iou=0.5,
    show=False,
    save=True,
    project='inference_output',
    name='demo'
)

# 可视化
for r in results:
    im_array = r.plot()
    im = im_array[:, :, ::-1]  # BGR to RGB
    cv2.imshow('Tank Detection', im)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. 视频或摄像头推理

```python
# 视频文件
model.predict(source='drone_video.mp4', show=True, save=True)

# 实时摄像头
model.predict(source=0, show=True, conf=0.4)
```

### 3. 批量推理

```python
model.predict(
    source='tank_dataset/images/val',
    save=True,
    project='batch_infer',
    name='val_results'
)
```

---

## ✅ 八、模型导出（用于部署）

```python
# 导出为 ONNX（通用部署）
model.export(format='onnx', dynamic=True, opset=13)

# 导出为 TensorRT（NVIDIA GPU 高性能）
# model.export(format='engine', device=0)

# 导出为 TorchScript
model.export(format='torchscript')
```

---

## ✅ 九、可视化训练过程

训练完成后查看：
- `runs/tank/yolov8m_tank_detection/results.png` → mAP、loss 曲线
- `confusion_matrix.png` → 分类准确性
- `val_batch*.jpg` → 检测效果示例

启动 TensorBoard：
```bash
tensorboard --logdir runs/tank
```

---

## ✅ 十、总结

| 步骤 | 内容 |
|------|------|
| 🧰 环境 | CUDA + Ultralytics YOLOv8 |
| 📁 数据 | YOLO 格式，`data.yaml` 配置 |
| 🏃‍♂️ 训练 | `model.train()` 高级调参 |
| 🔍 推理 | 图像、视频、摄像头支持 |
| 📊 评估 | mAP@0.5, mAP@0.5:0.95 |
| 🚀 部署 | ONNX / TensorRT 导出 |

---

📌 **应用场景**：
- 无人机战场感知
- 边防监控系统
- 军事演习智能分析
- 遥感图像自动解译

需要我帮你生成：
- **PR曲线绘制脚本**
- **混淆矩阵热力图**
- **FP/TP/FN 分析工具**
- **Web界面部署（Flask）**

欢迎继续提问！祝你训练顺利！🎯
