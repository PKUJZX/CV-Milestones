import sys
import os
import einops
import torch
import numpy as np
import cv2
from PIL import Image

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from cldm import ControlLDM 
from safetensors.torch import load_file as load_safetensors

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    # This is the new logic
    if ckpt.endswith(".safetensors"):
        sd = load_safetensors(ckpt)
    else:
        sd = torch.load(ckpt, map_location="cpu")
    
    # The key to load depends on the model version
    if "state_dict" in sd:
        pl_sd = {"state_dict": sd["state_dict"]}
    else:
        pl_sd = {"state_dict": sd}

    # ... the rest of the function stays the same from here
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(pl_sd["state_dict"], strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def process_image(input_image, low_threshold, high_threshold):
    """
    处理输入图像以生成 Canny 边缘图
    """
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)

    img = cv2.resize(input_image, (512, 512), interpolation=cv2.INTER_AREA)
    
    # 获取 Canny 边缘图
    canny_edges = cv2.Canny(img, low_threshold, high_threshold)
    
    # 扩展维度以匹配模型输入
    canny_edges = canny_edges[:, :, None]
    canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)
    
    # 转换为 torch 张量
    hint = torch.from_numpy(canny_edges.copy()).float().cuda() / 255.0
    hint = rearrange(hint, 'h w c -> 1 c h w')
    
    return hint, img


def main():
    # --- 配置模型路径 ---
    # 请确保这些路径是正确的
    base_model_path = './models/v1-5-pruned-emaonly.safetensors'
    controlnet_path = './models/control_sd15_canny.pth'
    config_path = './models/cldm_v15.yaml'

    # --- 加载模型 ---
    # 首先加载基础的 Stable Diffusion 模型权重
    model = load_model_from_config(OmegaConf.load(config_path), base_model_path)
    # 然后加载 ControlNet 的权重
    controlnet_sd = torch.load(controlnet_path, map_location='cpu')
    model.load_state_dict(controlnet_sd, strict=False)
    print("ControlNet 权重加载成功。")
    
    # --- 初始化采样器 ---
    ddim_sampler = DDIMSampler(model)

    # --- 推理参数 ---
    input_image_path = './test_imgs/dog.png' # 你的输入图片
    prompt = "A golden retriever dog running on the beach" # 你的文本提示
    
    # Canny 边缘检测的阈值
    low_threshold = 100
    high_threshold = 200
    
    # 生成参数
    num_samples = 1
    image_resolution = 512
    ddim_steps = 50
    guidance_scale = 9.0
    seed = 42
    eta = 0.0
    a_prompt = "best quality, extremely detailed" # 正面提示词增强
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality" # 负面提示词

    # --- 准备输入 ---
    seed_everything(seed)

    # 检查输入图片是否存在
    if not os.path.exists(input_image_path):
        print(f"错误: 输入图片未找到于 '{input_image_path}'")
        return

    input_image = Image.open(input_image_path).convert("RGB")
    
    # 处理图像以获得 Canny 边缘图作为 "hint"
    hint, source_image_resized = process_image(input_image, low_threshold, high_threshold)

    # --- 准备条件 ---
    cond = {
        "c_concat": [hint],
        "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
    }
    un_cond = {
        "c_concat": [hint],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
    }
    shape = (4, image_resolution // 8, image_resolution // 8) # 隐空间形状

    # --- 运行采样 ---
    print("开始生成图像...")
    model.low_vram_shift(is_diffusing=True) # 如果 VRAM 有限，可以进行优化

    samples, intermediates = ddim_sampler.sample(
        S=ddim_steps,
        conditioning=cond,
        batch_size=num_samples,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=un_cond,
        eta=eta
    )
    
    model.low_vram_shift(is_diffusing=False)

    # --- 解码并保存图像 ---
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    
    # 将原图、Canny边缘图和生成结果拼接在一起显示
    hint_display = (rearrange(hint, '1 c h w -> h w c').cpu().numpy() * 255).astype(np.uint8)
    final_result = np.concatenate([source_image_resized, hint_display, results[0]], axis=1)

    output_path = f'./output_dog_seed{seed}.png'
    Image.fromarray(final_result).save(output_path)
    print(f"生成完成！图像已保存至 {output_path}")


if __name__ == "__main__":
    # 确保模型文件夹存在
    if not os.path.exists('./models'):
        os.makedirs('./models')
        print("创建 './models' 文件夹。请将模型文件放入其中。")

    # 确保测试图片文件夹存在
    if not os.path.exists('./test_imgs'):
        os.makedirs('./test_imgs')
        print("创建 './test_imgs' 文件夹。请将测试图片放入其中。")
    
    main()