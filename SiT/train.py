import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from copy import deepcopy
from time import time
import os

from models import create_SiT_model
from transport import Transport
from diffusers.models import AutoencoderKL
from config import Config as args
from utils import update_ema, requires_grad, center_crop_arr


def main():
    # 1. 环境设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # 创建用于保存检查点和样本的目录
    checkpoint_dir = "checkpoints"
    sample_dir = "samples"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 2. 模型、VAE和Transport设置
    latent_size = args.image_size // 8
    
    # 创建SiT模型
    model = create_SiT_model(
        model_name=args.model,
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # 创建EMA模型
    ema = deepcopy(model).to(device)
    requires_grad(ema, False) 
    
    # 加载预训练的VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # 初始化Transport
    transport = Transport(train_eps=1e-5)

    # 3. 优化器和数据加载器
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # 创建数据集和加载器 
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        pin_memory=True,
        drop_last=True
    )

    # 4. 训练循环
    update_ema(ema, model, decay=0) 
    model.train() # 设置模型为训练模式
    ema.eval()    

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    for epoch in range(args.epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # 将图像编码到潜在空间
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            # 计算损失
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            # 优化步骤
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # 更新EMA模型
            update_ema(ema, model)

            # 5. 日志记录和保存
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            # 定期打印日志
            if train_steps % 100 == 0: 
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                print(f"(Step={train_steps:07d}) Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                # 重置计时器和计数器
                running_loss = 0
                log_steps = 0
                start_time = time()

            # 定期保存检查点
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()