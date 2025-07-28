import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

from sam import Sam
from models.image_encoder import ImageEncoderViT
from models.prompt_encoder import PromptEncoder
from models.mask_decoder import MaskDecoder
from models.transformer import TwoWayTransformer
from predict import SamPredictor

# --- 辅助函数 ---
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    if coords is None: return
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    if box is None: return
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 设置默认值
    IMAGE_PATH = "image.png"
    MODEL_PATH = "sam_vit_b.pth"

    # 2. 设置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points", type=int, nargs='+', help="点提示的坐标，格式为 x1 y1 x2 y2 ..."
    )
    parser.add_argument(
        "--labels", type=int, nargs='+', help="点提示的标签 (1=前景, 0=背景)，数量必须与点的数量一致。"
    )
    parser.add_argument(
        "--box", type=int, nargs=4, help="框提示的坐标，格式为 x1 y1 x2 y2 (左上角和右下角)。"
    )
    
    args = parser.parse_args()

    # 4. 构建 SAM 模型
    sam_model = Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, img_size=1024, mlp_ratio=4, num_heads=12, patch_size=16,
            qkv_bias=True, use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=256
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3, transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)
        ),
        pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]
    )

    # 5. 加载预训练权重
    with open(MODEL_PATH, "rb") as f:
        state_dict = torch.load(f)
    sam_model.load_state_dict(state_dict)

    # 6. 设置设备并初始化预测器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model.to(device)
    predictor = SamPredictor(sam_model)

    # 7. 加载图像并计算嵌入
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # 8. 准备提示
    input_points = np.array(args.points).reshape(-1, 2) if args.points else None
    input_labels = np.array(args.labels) if args.labels else None
    input_box = np.array(args.box) if args.box else None

    # 9. 进行预测
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        multimask_output=True,
    )

    # 10. 可视化结果
    # 遍历每一个预测出的掩码和其对应的分数
    scores, masks = zip(*sorted(list(zip(scores, masks)), key=lambda x: x[0], reverse=True))
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 12))
        show_mask(mask, plt.gca())
        if args.points is not None:
            show_points(input_points, input_labels, plt.gca(), marker_size=375)
        if args.box is not None:
            show_box(input_box, plt.gca())
        plt.title(f"Result {i+1}   |   Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()