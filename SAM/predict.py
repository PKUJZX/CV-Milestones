import numpy as np
import torch

from typing import Optional, Tuple

from sam import Sam
from utils import ResizeLongestSide


class SamPredictor:
    """
    使用 SAM 高效地为图像生成掩码。

    该预测器首先使用 `set_image` 方法为图像计算图像嵌入，
    然后可以根据不同的提示（如点或框）重复、高效地预测掩码。
    """

    def __init__(self, sam_model: Sam) -> None:
        """
        初始化 SamPredictor。

        参数:
          sam_model (Sam): 用于掩码预测的 SAM 模型实例。
        """
        super().__init__()
        self.model = sam_model
        # 初始化图像变换器，用于将图像调整到模型期望的尺寸
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """
        为提供的图像计算图像嵌入，以便后续使用 'predict' 方法进行掩码预测。

        参数:
          image (np.ndarray): 用于计算掩码的图像。期望是 HWC uint8 格式，
            像素值在 [0, 255] 范围内。
          image_format (str): 图像的颜色格式，可选值为 'RGB' 或 'BGR'。
        """
        assert image_format in ["RGB", "BGR"], f"图像格式必须是 'RGB' 或 'BGR'，但提供的是 {image_format}。"
        # 如果图像格式与模型期望的格式不符，则进行转换
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # 将图像变换为模型期望的格式
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        # 将 HWC 格式转换为 CHW，并增加一个批次维度 (BCHW)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(self, transformed_image: torch.Tensor, original_image_size: Tuple[int, ...]) -> None:
        """
        为已变换的图像张量计算图像嵌入。

        此方法期望输入图像已经被 `ResizeLongestSide` 变换处理过。

        参数:
          transformed_image (torch.Tensor): 形状为 1x3xHxW 的输入图像张量。
          original_image_size (tuple(int, int)): 变换前原始图像的尺寸 (H, W)。
        """
        img_size = self.model.image_encoder.img_size
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(transformed_image.shape[2:]) == img_size
        ), f"输入图像必须是 BCHW 格式，且最长边为 {img_size}。"

        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])

        # 预处理图像并计算图像嵌入
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用当前设置的图像，根据输入的提示预测掩码。

        参数:
          point_coords (np.ndarray | None): Nx2 的点坐标数组，格式为 (X, Y)。
          point_labels (np.ndarray | None): 长度为 N 的点标签数组。1 表示前景点，0 表示背景点。
          box (np.ndarray | None): 长度为 4 的框坐标数组，格式为 (X1, Y1, X2, Y2)。
          mask_input (np.ndarray | None): 低分辨率的掩码输入，通常来自上一次预测。形状为 1xHxW (H=W=256)。
          multimask_output (bool): 如果为 True，模型将返回三个掩码。对于模糊提示（如单个点），
            这通常能产生更好的结果。
          return_logits (bool): 如果为 True，返回未二值化的 logits，而不是二进制掩码。

        返回:
          (np.ndarray): C x H x W 格式的输出掩码，C 是掩码数量，(H, W) 是原始图像尺寸。
          (np.ndarray): 长度为 C 的数组，包含模型对每个掩码质量的预测分数。
          (np.ndarray): C x H x W (H=W=256) 的低分辨率 logits，可作为下一次迭代的 `mask_input`。
        """
        if not self.is_image_set:
            raise RuntimeError("在预测掩码之前，必须使用 .set_image() 设置图像。")

        # 将 NumPy 格式的提示转换为 PyTorch 张量
        coords_torch, labels_torch, box_torch, mask_input_torch = self._preprocess_prompts(
            point_coords, point_labels, box, mask_input
        )

        # 调用核心预测逻辑
        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        # 将结果从 PyTorch 张量转换为 NumPy 数组
        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _preprocess_prompts(
        self,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        mask_input: Optional[np.ndarray],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """将 NumPy 格式的提示转换为 PyTorch 张量。"""
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        
        # 处理点提示
        if point_coords is not None:
            assert point_labels is not None, "如果提供了 point_coords，则必须提供 point_labels。"
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        
        # 处理框提示
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
            
        # 处理掩码提示
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
            
        return coords_torch, labels_torch, box_torch, mask_input_torch

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        被 `predict` 方法调用，使用 PyTorch 张量作为提示进行预测的核心逻辑。
        """
        if not self.is_image_set:
            raise RuntimeError("在预测掩码之前，必须使用 .set_image() 设置图像。")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # 1. 编码提示
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # 2. 使用掩码解码器进行预测
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # 3. 将掩码上采样到原始图像分辨率
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        # 4. 根据需要对掩码进行二值化处理
        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        返回当前设置图像的嵌入。

        嵌入形状为 1xCxHxW，其中 C 是嵌入维度 (通常为 256)，
        (H, W) 是 SAM 嵌入的空间维度 (通常为 64x64)。
        """
        if not self.is_image_set:
            raise RuntimeError("必须先使用 .set_image() 设置图像才能获取嵌入。")
        assert self.features is not None, "如果已设置图像，则特征必须存在。"
        return self.features

    @property
    def device(self) -> torch.device:
        """获取模型所在的设备 (CPU or GPU)"""
        return self.model.device

    def reset_image(self) -> None:
        """重置当前设置的图像和相关状态。"""
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None