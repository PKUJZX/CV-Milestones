官方代码链接：[lllyasviel/ControlNet: Let us control diffusion models!](https://github.com/lllyasviel/ControlNet)

#### 1 安装依赖

```
pip install torch torchvision opencv-python einops pytorch-lightning transformers omegaconf safetensors open_clip_torch xformers
```

#### 2 下载预训练模型

**下载 Stable Diffusion V1.5**:

- 请从 [stable-diffusion-v1-5/stable-diffusion-v1-5 at main](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) 下载 `v1-5-pruned-emaonly.safetensors`。

**下载 ControlNet 模型**:

- 请从[官方 ControlNet 仓库](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) 下载 ControlNet 模型（`control_sd15_canny.pth`）。

将这两个文件放入 `models` 文件夹中。

#### 3 运行代码

```
python infer.py
```

首先对于输入图像( `test_imgs/dog.png` )，生成 Canny 边缘图。

该 Canny 边缘图和 `Prompt` (如"A golden retriever dog running on the beach") 被输入给 ControlNet ,生成最终图像。

<img width="1536" height="512" alt="image" src="https://github.com/user-attachments/assets/c9ba1bd1-0651-40fc-bec2-a81e64d5bd1d" />


您也可以使用自己的图片和Prompt，例如：

```
python infer.py \
    --input_path './path/to/your/input_image.png' \
    --prompt "A man in a spacesuit is on the moon" \
    --output_path 'results/moon_astronaut.png' 
```

