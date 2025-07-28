import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from models.image_encoder import ImageEncoderViT
from models.mask_decoder import MaskDecoder
from models.prompt_encoder import PromptEncoder


class Sam(nn.Module):
    """
    SAM 模型：根据图像和输入提示（prompt）预测物体的掩码（mask）。

    该模型集成了图像编码器、提示编码器和掩码解码器，
    实现了端到端的分割功能。
    """
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        初始化 Sam 模型。

        参数:
          image_encoder (ImageEncoderViT): 用于将图像编码为嵌入向量的主干网络，
            这使得高效的掩码预测成为可能。
          prompt_encoder (PromptEncoder): 用于编码各种类型的输入提示。
          mask_decoder (MaskDecoder): 根据图像嵌入和编码后的提示来预测掩码。
          pixel_mean (list(float)): 用于归一化输入图像像素的均值。
          pixel_std (list(float)): 用于归一化输入图像像素的标准差。
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        
        # 将像素均值和标准差注册为模型的缓冲区（buffer），这样它们会随着模型移动（例如，到 GPU），
        # 但不会被视为模型参数进行训练。
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        """获取模型所在的设备（如 'cpu' 或 'cuda'）。"""
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        根据提供的图像和提示，端到端地预测掩码。
        如果无法预先知道所有提示，建议使用 SamPredictor 而不是直接调用此方法。

        参数:
          batched_input (list(dict)): 一个包含多个输入图像信息的列表。每个元素是一个字典，
            包含以下键（如果某个提示不存在，可以省略对应的键）：
              'image': (torch.Tensor) 格式为 3xHxW 的图像张量，已经过预处理。
              'original_size': (tuple(int, int)) 图像预处理前的原始尺寸 (H, W)。
              'point_coords': (torch.Tensor) 点提示坐标，形状为 BxNx2。
              'point_labels': (torch.Tensor) 点提示的标签，形状为 BxN。
              'boxes': (torch.Tensor) 边界框提示，形状为 Bx4。
              'mask_inputs': (torch.Tensor) 掩码提示，形状为 Bx1xHxW。
          multimask_output (bool): 模型是否应预测多个消歧的掩码（True），还是返回单个掩码（False）。

        返回:
          (list(dict)): 一个包含多个输出结果的列表，每个元素对应一个输入图像，
            是一个包含以下键的字典：
              'masks': (torch.Tensor) 预测的二进制掩码，形状为 BxCxHxW。
              'iou_predictions': (torch.Tensor) 模型预测的掩码质量（IoU），形状为 BxC。
              'low_res_logits': (torch.Tensor) 低分辨率的 logits，形状为 BxCxHxW (H=W=256)。
        """
        # 1. 预处理所有图像并批量送入图像编码器
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        # 2. 针对每个图像和其对应的提示，独立地进行处理
        outputs = [
            self._process_single_image(
                image_record,
                embedding,
                multimask_output,
            )
            for image_record, embedding in zip(batched_input, image_embeddings)
        ]
        
        return outputs

    def _process_single_image(
        self,
        image_record: Dict[str, Any],
        image_embedding: torch.Tensor,
        multimask_output: bool,
    ) -> Dict[str, torch.Tensor]:
        """处理单个图像的分割预测逻辑。"""
        # 1. 准备提示输入
        points = (image_record["point_coords"], image_record["point_labels"]) if "point_coords" in image_record else None
        
        # 2. 编码提示，生成稀疏和密集嵌入
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=image_record.get("boxes"),
            masks=image_record.get("mask_inputs"),
        )
        
        # 3. 使用解码器，根据图像和提示嵌入预测掩码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding.unsqueeze(0),
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        # 4. 对掩码进行后处理，恢复到原始图像尺寸
        masks = self.postprocess_masks(
            masks=low_res_masks,
            input_size=image_record["image"].shape[-2:],
            original_size=image_record["original_size"],
        )
        
        # 5. 应用阈值，生成最终的二进制掩码
        return {
            "masks": masks > self.mask_threshold,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        移除填充并将掩码上采样至原始图像尺寸。

        参数:
          masks (torch.Tensor): 来自掩码解码器的批量掩码，格式为 BxCxHxW。
          input_size (tuple(int, int)): 输入到模型的图像尺寸 (H, W)，用于移除填充。
          original_size (tuple(int, int)): 调整大小前的原始图像尺寸 (H, W)。

        返回:
          (torch.Tensor): 批量掩码，格式为 BxCxHxW，其中 (H, W) 是 original_size。
        """
        # 将掩码上采样到图像编码器期望的输入尺寸 (e.g., 1024x1024)
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # 移除因填充（padding）而产生的多余部分
        masks = masks[..., : input_size[0], : input_size[1]]
        # 将掩码再次上采样到原始图像的尺寸
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """归一化像素值并将图像填充为正方形输入。"""
        # 归一化颜色
        x = (x - self.pixel_mean) / self.pixel_std

        # 填充至正方形
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x