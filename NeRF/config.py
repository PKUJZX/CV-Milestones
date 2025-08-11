# ==========================================================================================
# NeRF 项目配置文件
# 所有实验参数、路径和设置都集中在此处，方便统一管理和修改。
# ==========================================================================================

config = {
    # 核心参数
    "expname": 'lego_exp_final',
    "datadir": './data/nerf_synthetic/lego',
    
    # 目录和保存选项
    "basedir": './logs/',
    "i_print": 100,       # 每隔i_print步，在控制台打印一次训练状态
    "i_weights": 10000,   # 每隔i_weights步，保存一次模型权重
    "i_video": 50000,     # 每隔i_video步，渲染并保存一次测试视频
    "i_testset": 50000,   # 每隔i_testset步，在测试集上渲染并保存结果

    # 网络结构参数
    "netdepth": 8,        # "粗糙"网络深度
    "netwidth": 256,      # "粗糙"网络宽度
    "netdepth_fine": 8,   # "精细"网络深度
    "netwidth_fine": 256, # "精细"网络宽度
    
    # 训练过程参数
    "N_rand": 1024,       # 每批光线的数量 (batch size)
    "lrate": 5e-4,        # 学习率
    "lrate_decay": 500,   # 学习率衰减步长 (单位: k steps)
    "chunk": 1024 * 32,   # 一次送入GPU处理的光线数量，用于防止OOM
    "netchunk": 1024 * 64, # 一次送入网络处理的点的数量，用于防止OOM
    "no_reload": False,   # 是否不加载已有的权重，从头训练
    "ft_path": None,      # 指定特定的权重文件路径进行加载

    # 渲染参数
    "N_samples": 64,      # 每条光线的粗采样点数
    "N_importance": 128,  # 每条光线的精细采样点数 (分层采样)
    "perturb": 1.0,       # 是否在采样时加入随机扰动 (0表示关闭)
    "use_viewdirs": True, # 是否使用视角方向作为输入
    "raw_noise_std": 0.0, # 给原始输出添加的噪声标准差，用于正则化
    
    # 位置编码参数
    "i_embed": 0,         # 编码方案索引 (0 for default positional encoding, -1 for none)
    "multires": 10,       # 位置编码的频率数量 (for xyz)
    "multires_views": 4,  # 视角方向编码的频率数量 (for view directions)

    # 数据集特定参数 (Blender)
    "dataset_type": 'blender',
    "testskip": 8,        # 加载测试/验证集时，每隔testskip张图片取一张
    "white_bkgd": True,   # 数据集图像是否为白色背景
    "half_res": True,     # 是否将图像分辨率减半
    
    # 渲染模式选项
    "render_only": False, # 是否只执行渲染而不进行训练
    "render_test": False, # 渲染模式下，是否渲染测试集(否则渲染预设的环绕路径)
    "render_factor": 0,   # 渲染降采样因子，0表示不降采样
    
    # 训练总迭代次数
    "N_iters": 200001
}
