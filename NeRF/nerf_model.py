import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================================================================
# 位置编码 (Positional Encoding)
# NeRF的核心创新之一。它将低维的坐标输入(如x,y,z)映射到高维空间，
# 使得神经网络能够学习到场景中更高频率的细节变化，对于渲染清晰图像至关重要。
# ==========================================================================================

class Embedder:
    """
    将输入进行位置编码。
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        
        # 选项：是否直接包含原始输入
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        # 生成不同频率的频带
        if self.kwargs['log_sampling']:
            # 按对数尺度采样频率，更密集地覆盖低频区域
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # 按线性尺度采样频率
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        # 对每个频率应用周期函数(sin, cos)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        """执行编码"""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """
    获取一个位置编码器函数和其输出维度。

    参数:
        multires (int): 频率数量的log2值。L in the paper.
        i (int): 编码方案的索引。0表示默认的位置编码，-1表示无编码。

    返回:
        embed_fn (function): 编码函数。
        input_ch (int): 编码后的输出维度。
    """
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# ==========================================================================================
# NeRF 模型
# ==========================================================================================

class NeRF(nn.Module):
    """
    NeRF 模型的神经网络结构。
    """
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        初始化NeRF模型。
        参数:
            D (int): 网络深度（层数）。
            W (int): 每层的宽度（通道数）。
            input_ch (int): 位置(pts)输入的通道数。
            input_ch_views (int): 视角方向(views)输入的通道数。
            output_ch (int): 输出通道数 (RGB+Alpha)。
            skips (list of int): 需要进行残差连接的层索引。
            use_viewdirs (bool): 是否使用视角方向作为输入。
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # 用于处理空间位置(pts)的MLP。
        # 包含8个线性层。
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # 用于处理视角方向(views)的MLP。
        # 包含1个线性层。
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            # 预测体积密度 (alpha) 的线性层
            self.alpha_linear = nn.Linear(W, 1)
            # 提取特征向量的线性层
            self.feature_linear = nn.Linear(W, W)
            # 预测颜色 (RGB) 的线性层
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            # 如果不使用视角方向，则直接输出RGB+Alpha
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        前向传播。
        参数:
            x (Tensor): 输入张量，包含了位置编码后的坐标和视角方向。
        返回:
            outputs (Tensor): 输出张量，包含了预测的RGB颜色和体积密度alpha。
        """
        # 1. 将输入分割为位置(pts)和视角(views)
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        
        # 2. 将位置信息传入第一个MLP块
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # 在指定层(skip connection)处，将原始输入再次拼接进来
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # 3. 基于位置信息，预测出体积密度和特征向量
        if self.use_viewdirs:
            # 3a. 预测 alpha (体积密度)，它只依赖于空间位置
            alpha = self.alpha_linear(h)
            
            # 3b. 提取特征向量
            feature = self.feature_linear(h)
            
            # 3c. 将特征向量和视角方向拼接
            h = torch.cat([feature, input_views], -1)
            
            # 4. 将拼接后的向量传入第二个MLP块，以预测视角相关的颜色
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            # 5. 预测 RGB 颜色
            rgb = self.rgb_linear(h)
            
            # 6. 将颜色和体积密度拼接为最终输出
            outputs = torch.cat([rgb, alpha], -1)
        else:
            # 如果不使用视角，则直接从第一个MLP块的结果中输出
            outputs = self.output_linear(h)

        return outputs
