import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from models import create_SiT_model
from transport import Transport, Sampler
from config import Config as args 
from utils import find_model


def main():
    # 1. 环境和设备设置
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False) 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载模型、VAE 和采样器
    latent_size = args.image_size // 8
    
    # 创建SiT模型并移动到指定设备
    model = create_SiT_model(
        model_name=args.model,
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)
    
    # 加载预训练的模型检查点 (find_model会自动处理EMA权重)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # 初始化Transport和ODE采样器
    transport = Transport(sample_eps=1e-3)
    sampler = Sampler(transport)

    sample_fn = sampler.get_sampler_fn(
        num_steps=args.num_sampling_steps,
        atol=args.atol,
        rtol=args.rtol
    )
            
    # 加载VAE模型，用于将潜在表示解码为图像
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # 3. 准备采样输入 (噪声和条件)
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)

    # 创建初始噪声 z 和类别标签 y
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # 设置模型的前向传播函数和参数，以支持CFG
    if args.cfg_scale > 1.0:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        y = torch.cat([y, y_null], 0) 
        
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_fn = model.forward_with_cfg 
    else:
        model_kwargs = dict(y=y)
        model_fn = model.forward

    # 4. 执行采样并保存结果
    # 调用ODE求解器进行采样，得到潜在空间中的样本
    samples_latent = sample_fn(z, model_fn, **model_kwargs)[-1]
    
    # 如果使用了CFG，只取有条件生成的那部分结果
    if args.cfg_scale > 1.0:
        samples_latent, _ = samples_latent.chunk(2, dim=0)
    
    # 使用VAE解码器将潜在样本转换为图像
    with torch.no_grad():
        samples_image = vae.decode(samples_latent / 0.18215).sample

    # 保存生成的图像
    save_image(samples_image, args.output_file, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {args.output_file}")

if __name__ == "__main__":
    main()
