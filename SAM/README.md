官方代码库链接：[facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.](https://github.com/facebookresearch/segment-anything)

#### 文件结构说明

```
SAM/
├── models/
│   ├── image_encoder.py   # 图像编码器
│   ├── mask_decoder.py    # 掩码解码器
│   ├── prompt_encoder.py  # 提示编码器
│   ├── transformer.py     # Transformer 模块
│   └── common.py          # 通用模块（如 MLP Block）
├── utils.py               # 实用工具（如图像缩放）
├── sam.py                 # SAM 主模型
├── predict.py             # 预测器类，用于简化单张图像的预测流程
├── run.py                 # 主运行脚本
└── image.png              # 示例输入图像
```

**建议重点理解**：predict.py & sam.py

#### 1. 创建环境

```
conda create --name SAM python=3.11
conda activate SAM
```

#### 2. 安装所需的库

```
pip install torch torchvision numpy opencv-python matplotlib
```

#### 3. 下载模型权重

您可以从[这里](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)下载 **ViT-B** 模型的权重。

下载后，将文件重命名为 `sam_vit_b.pth` 并将其与 `run.py` 放在同一个文件夹下。

#### 3. 执行代码

示例图像：

<img src="C:\Users\admin'\AppData\Roaming\Typora\typora-user-images\image-20250729201832256.png" alt="image-20250729201832256" style="zoom:50%;" />

##### **示例 1: 使用点提示进行分割**

假设您想分割图像上坐标为 (700, 400) 的物体，这是一个前景点。

```
python run.py --points 700 400 --labels 1
```

<img src="C:\Users\admin'\AppData\Roaming\Typora\typora-user-images\image-20250729203020388.png" alt="image-20250729203020388" style="zoom:50%;" />

##### **示例 2: 使用边界框提示进行分割**

假设您想分割一个由左上角 (100, 200) 和右下角 (300, 500) 定义的边界框内的物体。

```
python run.py --box 100 200 300 500
```

<img src="C:\Users\admin'\AppData\Roaming\Typora\typora-user-images\image-20250729203139411.png" alt="image-20250729203139411" style="zoom:50%;" />
