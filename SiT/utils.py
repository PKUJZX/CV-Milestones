import torch
from torchvision.datasets.utils import download_url
from collections import OrderedDict
from PIL import Image
import numpy as np
import os

# 预训练模型的名称列表
pretrained_models = {'SiT-XL-2-256x256.pt'}

def find_model(model_name):
    """
    查找并加载模型检查点。
    如果 model_name 是一个预定义的模型名称，则从网络下载。
    否则，将其视为本地文件路径加载。
    """
    if model_name in pretrained_models:  
        return download_model(model_name)
    else:  
        if not os.path.isfile(model_name):
            raise FileNotFoundError(f'在 {model_name} 处找不到SiT检查点')
        
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  
            checkpoint = checkpoint["ema"]
        return checkpoint

def download_model(model_name):
    """
    从指定的URL下载预训练的SiT模型。
    """
    # 确保本地存在保存模型的目录
    local_dir = 'pretrained_models'
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, model_name)
    
    if not os.path.isfile(local_path):
        print(f"Downloading {model_name}...")
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
        download_url(web_path, local_dir, filename=model_name)
        print("Download complete.")

    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    将输入的PIL图像进行高质量的中心裁剪，以匹配模型所需的输入尺寸。
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
