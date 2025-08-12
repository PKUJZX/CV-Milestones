官方代码链接：https://github.com/willisma/SiT

这个项目分为两个主要任务：**训练 (Training)** 和 **采样 (Sampling)**。

- **训练**：从零开始在一个大规模数据集（如 ImageNet）上训练你自己的 SiT 模型。
- **采样**：使用一个已经训练好的模型检查点（checkpoint）来生成新的图像。

对于初次尝试，建议您先从 **采样** 开始，因为它更简单，能快速看到结果。

#### 第一步：环境准备与依赖安装

```
pip install torch torchvision timm diffusers torchdiffeq Pillow numpy
```

#### **第二步：准备模型检查点**

该代码非常方便，可以自动下载预训练好的模型。根据 `utils.py` 文件，当您运行采样脚本时，它会检查 `pretrained_models` 目录。如果找不到指定的模型文件（`SiT-XL-2-256x256.pt`），它会自动从网上下载。

所以，您无需手动下载模型，只需确保您的网络连接正常。

#### 第三步：运行采样 (生成图像)

```
python sample.py
```

脚本会执行以下操作：

- 检查并下载预训练的 SiT 模型和 VAE 模型。
- 使用 `torchdiffeq` 求解器从随机噪声开始生成 8 个图像的潜在表示。
- 使用 VAE 解码器将这些潜在表示转换为图像。
- 最后，将生成的 8 张图像保存在一个名为 `sample.png` 的文件中。

<img width="1034" height="518" alt="image" src="https://github.com/user-attachments/assets/c3b3cbb6-602c-4692-b6a2-17f7f8c722bb" />

#### **第四步：运行训练 (可选，需要数据集和强大GPU)**

如果您想自己训练模型，请注意这将需要：

- 一个非常大的数据集，代码默认使用 ImageNet。
- 一块或多块拥有大显存的高性能 GPU。
- 较长的训练时间（可能需要几天甚至几周）。

1. **准备数据集**:

   - 下载 ImageNet 数据集。
   - 将其解压到一个文件夹中，该文件夹应包含 `train` 子目录，`train` 里面是按类别分的子文件夹。

2. **修改配置 (`config.py`)**:

   - 打开 `config.py` 文件，将 `data_path` 的值修改为您真实的 ImageNet `train` 文件夹的路径。

3. **修改训练脚本**:

   本项目主要目的是便于学习者理解，因此去掉了分布式训练(DDP)，但若你想从头开始训练模型，推荐自己修改脚本添加DDP功能，以加速训练。

4. **运行训练脚本**:

   ```
   python train.py
   ```

   脚本会开始训练过程：

   - 它会加载数据集，创建模型、优化器等。
   - 然后进入漫长的训练循环，定期在终端打印出损失（Loss）和训练速度。
   - 根据 `config.py` 中的 `ckpt_every` 设置，脚本会定期在 `checkpoints` 目录下保存模型的检查点文件（`.pt`）。






