import einops
import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    """
    一个被修改过的 U-Net 模型，能够接收并应用来自 ControlNet 的控制信号。

    这个类继承自原始的 UNetModel，但重写了其 forward 方法，
    以便在 U-Net 的解码器（上采样）阶段将 ControlNet 的输出添加到跳跃连接（skip connections）中。
    """

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor = None, context: torch.Tensor = None, control: list = None, only_mid_control: bool = False, **kwargs) -> torch.Tensor:
        """
        前向传播过程。

        Args:
            x (torch.Tensor): 带噪声的输入隐空间张量，形状为 (B, C, H, W)。
            timesteps (torch.Tensor): 当前的扩散时间步。
            context (torch.Tensor): 来自文本编码器的条件上下文。
            control (list, optional): 一个包含 ControlNet 输出特征图的列表。Defaults to None.
            only_mid_control (bool, optional): 是否只应用中间块的控制。Defaults to False.

        Returns:
            torch.Tensor: U-Net 预测出的噪声。
        """
        # 存储编码器各层的输出（跳跃连接）
        hs = []
        
        # 冻结原始 U-Net 的编码器和中间块，不计算梯度
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        # 将 ControlNet 的中间块输出应用到 U-Net 的中间块
        if control is not None:
            h += control.pop()

        # 遍历解码器的每一层
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                # 如果只使用中间控制或没有控制信号，则正常拼接跳跃连接
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                # 核心：将 ControlNet 的输出加到对应的跳跃连接上，再进行拼接
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    """
    ControlNet 的核心网络结构。

    它的架构是原始扩散模型 U-Net 编码器的一个可训练副本。
    它接收一个额外的条件输入（"hint"），并生成一系列控制信号（特征图），
    这些信号将被注入到主 U-Net 的解码器中。
    """
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        # 参数校验
        if use_spatial_transformer:
            assert context_dim is not None, '使用空间变换器时必须提供 context_dim (交叉注意力维度)'
        if context_dim is not None:
            assert use_spatial_transformer, '提供了 context_dim 但未使用空间变换器'
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, '必须设置 num_heads 或 num_head_channels'
        if num_head_channels == -1:
            assert num_heads != -1, '必须设置 num_heads 或 num_head_channels'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            assert len(num_res_blocks) == len(channel_mult), "num_res_blocks 列表长度应与 channel_mult 一致"
            self.num_res_blocks = num_res_blocks

        # --- 1. 时间步嵌入 (Time Embedding) ---
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # --- 2. 条件提示编码器 (Hint Encoder) ---
        # 这个模块专门用来处理输入的条件提示（如 Canny 边缘图）
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            # 使用零卷积初始化，确保训练初期不影响主模型
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        # --- 3. ControlNet 编码器主干 (Encoder Backbone) ---
        # 这个主干网络的结构与原始 U-Net 的编码器完全相同
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        # 零卷积层列表，与每个输出块对应
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        input_block_chans = [model_channels]
        current_channels = model_channels
        resolution_level = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        current_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                current_channels = mult * model_channels
                if resolution_level in attention_resolutions:
                    dim_head = current_channels // num_heads
                    layers.append(
                        AttentionBlock(
                            current_channels, use_checkpoint=use_checkpoint, num_heads=num_heads,
                            num_head_channels=dim_head, use_new_attention_order=use_new_attention_order
                        ) if not use_spatial_transformer else SpatialTransformer(
                            current_channels, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=False, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(current_channels))
                input_block_chans.append(current_channels)
            
            # 下采样块
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(current_channels, conv_resample, dims=dims))
                )
                self.zero_convs.append(self.make_zero_conv(current_channels))
                resolution_level *= 2
                input_block_chans.append(current_channels)

        # --- 4. 中间块 (Middle Block) ---
        self.middle_block = TimestepEmbedSequential(
            ResBlock(current_channels, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint),
            SpatialTransformer(
                current_channels, num_heads, current_channels // num_heads, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer, use_checkpoint=use_checkpoint
            ),
            ResBlock(current_channels, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint),
        )
        self.middle_block_out = self.make_zero_conv(current_channels)

    def make_zero_conv(self, channels):
        """创建一个权重和偏置都初始化为零的 1x1 卷积层。"""
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x: torch.Tensor, hint: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, **kwargs) -> list:
        """
        ControlNet 的前向传播。

        Args:
            x (torch.Tensor): U-Net 的输入，即带噪声的隐空间张量。
            hint (torch.Tensor): 条件控制输入，如 Canny 边缘图。
            timesteps (torch.Tensor): 当前扩散时间步。
            context (torch.Tensor): 文本条件上下文。

        Returns:
            list: 一个包含所有控制信号（特征图）的列表。
        """
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # 1. 处理条件提示
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []
        h = x.type(self.dtype)

        # 2. 遍历编码器主干
        for i, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            # 在第一个块之后，将处理过的提示注入网络
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            
            # 将每个块的输出通过对应的零卷积层，得到控制信号
            outs.append(zero_conv(h, emb, context))

        # 3. 处理中间块
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):
    """
    完整的、集成了 ControlNet 的隐空间扩散模型 (LDM)。

    这个类继承自 LatentDiffusion，并重写了其中的关键方法，
    以将 ControlNet 无缝集成到训练和推理流程中。
    """
    def __init__(self, control_stage_config, control_key, only_mid_control, sd_locked=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 从配置实例化 ControlNet
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.sd_locked = sd_locked # 控制是否锁定（冻结）原始 SD 模型的权重
        self.control_scales = [1.0] * 13

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        在单个去噪步骤中应用模型（ControlNet + 主 U-Net）。
        这是模型的核心逻辑所在。
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # 这是 ControlledUnetModel
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        # 如果没有提供条件控制（例如，在无分类器指导的负向传播中）
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            # 1. 调用 control_model 生成控制信号
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            
            # (可选) 对控制信号应用缩放因子
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            
            # 2. 将控制信号传入主 U-Net 进行预测
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        重写 get_input 以处理 ControlNet 的条件输入。
        """
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        
        # 从 batch 中获取 ControlNet 的条件控制输入 ("hint")
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        
        return x, dict(c_crossattn=[c], c_concat=[control])

    def configure_optimizers(self):
        """
        配置优化器。
        关键在于，我们只训练 ControlNet 的参数，而保持原始 SD 模型冻结。
        """
        lr = self.learning_rate
        # 只获取 control_model 的参数进行优化
        params = list(self.control_model.parameters())
        
        # 如果没有锁定 SD 模型，则也训练其输出层（可选的微调策略）
        if not self.sd_locked:
            print("训练 ControlNet 和 SD U-Net 解码器。")
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        else:
            print("只训练 ControlNet。")
            
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing: bool):
        """
        一个在低显存设备上进行优化的工具函数。
        在推理时，将不需要的模型移到 CPU，需要的移到 GPU，以节省显存。
        """
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
    
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0,
                   unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0 # 将控制图像反归一化以便可视化
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if sample:
            samples, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                         batch_size=N, ddim=use_ddim,
                                         ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat # 对于 ControlNet, 负向条件也使用相同的控制输入
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates