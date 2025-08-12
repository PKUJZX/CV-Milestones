import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


# 对输入特征 x 进行仿射变换（缩放和位移）
def modulate(x, shift, scale):
    """
    通过学习到的 shift 和 scale 参数来调整输入张量 x。
    这是 Adaptive LayerNorm (adaLN) 的核心操作。
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# 时间步 t 的嵌入器
class TimestepEmbedder(nn.Module):
    """
    将扩散过程中的时间步 t 转换为一个固定维度的向量嵌入。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        使用不同频率的正弦和余弦函数创建时间步 t 的位置编码。
        这是借鉴自 Transformer 的经典位置编码方法。
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # 将时间步 t 转换为频率编码，再通过 MLP 映射为最终嵌入
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# 类别标签 y 的嵌入器
class LabelEmbedder(nn.Module):
    """
    将类别标签 y 转换为一个固定维度的向量嵌入。
    支持在训练时通过 dropout 实现 Classifier-Free Guidance (CFG)。
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        # 嵌入表多一个位置用于存储 "无条件" (unconditional) 的嵌入
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        根据 dropout 概率，随机将一些标签替换为 "无条件" 标签。
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        # 在训练或强制丢弃时，应用 token_drop
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


# SiT 模型的核心块
class SiTBlock(nn.Module):
    """
    一个 SiT 块，即一个带有 Adaptive LayerNorm (adaLN) 的 Transformer 块。
    adaLN 使用条件信息 c (来自时间和类别) 来调整归一化层的参数。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # 学习从条件 c 生成 6 个调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # 从条件 c 生成 shift, scale, gate 参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # 第一个残差连接：自注意力模块
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # 第二个残差连接：MLP 模块
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# 模型的最后一层
class FinalLayer(nn.Module):
    """
    模型的最后一层，将 Transformer 块的输出映射回图像 patch 的维度。
    同样使用了 adaLN 进行特征调制。
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# Scalable Interpolant Transformer (SiT) 主模型
class SiT(nn.Module):
    """
    SiT 模型的主体结构。
    它将输入图像、时间步和类别标签作为输入，并输出去噪后的图像。
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # 模块定义
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # 位置编码，固定不可训练
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # SiT 块堆叠
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化模型权重。
        对不同模块使用特定的初始化策略。
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 初始化位置编码 (sin-cos)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化 PatchEmbed 权重
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 初始化标签嵌入和时间嵌入
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 初始化 SiT 块中的 adaLN 调制层
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 初始化 FinalLayer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        将 patch 序列重新组合成图像。
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        模型的前向传播。
        """
        x = self.x_embedder(x) + self.pos_embed  # 图像嵌入 + 位置编码
        t = self.t_embedder(t)                  # 时间嵌入
        y = self.y_embedder(y, self.training)  # 类别嵌入
        c = t + y                              # 融合时间和类别作为条件
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)             # 最终输出层
        x = self.unpatchify(x)                 # 还原为图像
        if self.learn_sigma:
            # 如果模型学习方差，则输出通道数加倍，这里只取一半作为预测的噪声
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        使用 Classifier-Free Guidance (CFG) 的前向传播。
        CFG 通过结合有条件和无条件模型的输出来增强生成效果。
        """
        # 将输入批次复制一份，一份用于有条件预测，一份用于无条件预测
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # 假设输出的前3个通道是预测的噪声
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # CFG 公式: unconditional + scale * (conditional - unconditional)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# --- 位置编码辅助函数 ---

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    生成 2D sin-cos 位置编码。
    :param embed_dim: 嵌入维度
    :param grid_size: 网格大小 (例如, 16 for 256x256 image with patch size 16)
    :return: (grid_size*grid_size, embed_dim) 的位置编码矩阵
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """从网格生成2D位置编码"""
    assert embed_dim % 2 == 0
    # 分别计算高和宽的位置编码
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    # 拼接高和宽的编码
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """从1D位置网格生成sin-cos编码"""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# --- 模型创建工厂函数 ---

def create_SiT_model(model_name: str, **kwargs) -> SiT:
    """
    根据模型名称 (如 'SiT-B/2') 创建一个 SiT 模型实例。
    """
    # 从模型名称中解析尺寸和 patch 大小
    model_size, patch_size_str = model_name.split('/')
    patch_size = int(patch_size_str)
    
    # 不同尺寸模型的配置
    size_configs = {
        'SiT-S': {'depth': 12, 'hidden_size': 384, 'num_heads': 6},
        'SiT-B': {'depth': 12, 'hidden_size': 768, 'num_heads': 12},
        'SiT-L': {'depth': 24, 'hidden_size': 1024, 'num_heads': 16},
        'SiT-XL': {'depth': 28, 'hidden_size': 1152, 'num_heads': 16},
    }
        
    config = size_configs[model_size]
    # 合并默认配置、传入的kwargs和从名称解析的patch_size
    final_config = {**config, **kwargs, "patch_size": patch_size}
    
    return SiT(**final_config)
