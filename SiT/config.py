class Config:
    # --- 基本配置  ---
    model = "SiT-XL/2"                  # 模型型号, e.g., SiT-S/2, SiT-B/2, SiT-L/2, SiT-XL/2
    image_size = 256                    # 图像尺寸 (必须是8的倍数)
    num_classes = 1000                  # 数据集中的类别总数 (ImageNet)
    vae = "ema"                         # VAE模型类型, 可选 "ema" 或 "mse"
    seed = 0                            # 随机种子, 用于复现结果

    # --- 训练配置 ---
    data_path = "/path/to/your/imagenet/train"  # 训练数据集的路径 (需要用户指定)
    epochs = 1400                       # 训练总轮数
    batch_size = 128                    # 批处理大小 (根据你的GPU显存调整)
    lr = 1e-4                           # 学习率
    ema_decay = 0.9999                  # EMA模型的衰减率
    ckpt_every = 50_000                 # 每隔多少训练步保存一次检查点
    
    # --- 采样/生成配置 ---
    ckpt = "SiT-XL-2-256x256.pt"         # 用于采样的模型检查点路径
    cfg_scale = 4.0                     # 无分类器指导的缩放因子 (推荐值: 3.0-7.0)
    num_sampling_steps = 50             # ODE采样的步数 (推荐值: 30-100)
    output_file = "sample.png"          # 生成样本的输出文件名

    # --- ODE求解器配置 ---
    atol = 1e-6                         # 绝对容忍度
    rtol = 1e-3                         # 相对容忍度
