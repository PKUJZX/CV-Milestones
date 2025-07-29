import math
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

# 从 utils.py 导入所有需要的辅助函数
from utils import (
    get_view_matrix, get_projection_matrix, transform_points, 
    quaternion_to_matrix, compute_cov2d, compute_color_from_sh
)

@dataclass
class PreprocessedData:
    """
    用于存储预处理后的高斯数据的容器，已按深度排序。
    """
    means2d: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    conics: np.ndarray
    radii: np.ndarray


class GaussianRasterizer:
    """
    一个实现了3D高斯溅射渲染流程的类。
    它将3D高斯场景数据和相机参数作为输入，渲染出2D图像。
    """

    def __init__(self, width: int = 700, height: int = 700):
        """
        初始化渲染器。
        Args:
            width (int): 输出图像的宽度。
            height (int): 输出图像的高度。
        """
        self.width = width
        self.height = height

    def forward(
        self,
        means3D: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        shs: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        cam_pos: np.ndarray,
        tan_fovx: float,
        tan_fovy: float,
        bg_color: np.ndarray = np.array([0.0, 0.0, 0.0])
    ) -> np.ndarray:
        """
        执行完整的前向渲染流程。
        """
        print("开始预处理高斯数据...")
        preprocessed_data = self._preprocess(
            means3D, opacities, scales, rotations, shs,
            view_matrix, proj_matrix, cam_pos,
            tan_fovx, tan_fovy
        )

        if preprocessed_data is None:
            print("没有高斯点在视锥体内，返回背景图像。")
            return np.tile(bg_color, (self.height, self.width, 1))

        print("开始渲染图像...")
        rendered_image = self._render(preprocessed_data, bg_color)
        return rendered_image

    def _preprocess(
        self,
        means3D: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        shs: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        cam_pos: np.ndarray,
        tan_fovx: float,
        tan_fovy: float
    ) -> PreprocessedData | None:
        """
        对所有3D高斯进行预处理，包括坐标变换、视锥体裁剪、计算2D协方差和颜色，并按深度排序。
        """
        view_proj_matrix = proj_matrix @ view_matrix

        # --- 1. 坐标变换与视锥体裁剪 ---
        p_view = transform_points(means3D, view_matrix)
        # 深度是相机前方-Z轴的距离，所以需要取反
        depths = -p_view[:, 2] 
        in_frustum_mask = depths > 0.2
        
        if not np.any(in_frustum_mask):
            return None

        culled_means3D = means3D[in_frustum_mask]
        culled_p_view = p_view[in_frustum_mask]
        culled_opacities = opacities[in_frustum_mask]
        culled_scales = scales[in_frustum_mask]
        culled_rotations = rotations[in_frustum_mask]
        culled_shs = shs[in_frustum_mask]
        culled_depths = depths[in_frustum_mask]

        p_hom = transform_points(culled_means3D, view_proj_matrix)
        p_w = p_hom[:, 3, np.newaxis]
        p_ndc = p_hom[:, :3] / p_w
        
        means2d = np.zeros((p_ndc.shape[0], 2))
        means2d[:, 0] = ((p_ndc[:, 0] + 1.0) * self.width - 1.0) * 0.5
        means2d[:, 1] = ((p_ndc[:, 1] + 1.0) * self.height - 1.0) * 0.5

        # --- 2. 计算协方差 ---
        R = quaternion_to_matrix(culled_rotations)
        S = np.zeros((R.shape[0], 3, 3))
        S[:, 0, 0] = culled_scales[:, 0]
        S[:, 1, 1] = culled_scales[:, 1]
        S[:, 2, 2] = culled_scales[:, 2]
        cov3D = R @ S @ S.transpose(0, 2, 1) @ R.transpose(0, 2, 1)

        focal_x = self.width / (2 * tan_fovx)
        focal_y = self.height / (2 * tan_fovy)
        conics = compute_cov2d(culled_p_view, focal_x, focal_y, cov3D, view_matrix)

        # --- 3. 计算颜色 ---
        view_dirs = culled_means3D - cam_pos
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)
        colors = compute_color_from_sh(culled_shs, view_dirs)

        # --- 4. 计算半径并排序 ---
        det = conics[:, 0] * conics[:, 2] - conics[:, 1] * conics[:, 1]
        mid = 0.5 * (conics[:, 0] + conics[:, 2])
        lambda1 = mid + np.sqrt(np.maximum(0.1, mid**2 - det))
        radii = np.ceil(3 * np.sqrt(lambda1))

        sort_indices = np.argsort(culled_depths)[::-1]
        
        return PreprocessedData(
            means2d=means2d[sort_indices],
            colors=colors[sort_indices],
            opacities=culled_opacities[sort_indices].flatten(),
            conics=conics[sort_indices],
            radii=radii[sort_indices]
        )

    def _render(self, data: PreprocessedData, bg_color: np.ndarray) -> np.ndarray:
        """
        使用“逐高斯”的方法进行渲染。
        """
        out_image = np.tile(bg_color, (self.height, self.width, 1))
        transmittance = np.ones((self.height, self.width))

        num_gaussians = data.means2d.shape[0]
        
        for i in tqdm(range(num_gaussians), desc="渲染高斯点"):
            if transmittance.max() < 1e-4:
                break

            mean = data.means2d[i]
            radius = int(data.radii[i])
            color = data.colors[i]
            conic = data.conics[i]
            opacity = data.opacities[i]
            
            min_x = max(0, int(mean[0] - radius))
            max_x = min(self.width, int(mean[0] + radius))
            min_y = max(0, int(mean[1] - radius))
            max_y = min(self.height, int(mean[1] + radius))

            if max_x <= min_x or max_y <= min_y:
                continue

            y_coords, x_coords = np.meshgrid(np.arange(min_y, max_y), np.arange(min_x, max_x), indexing='ij')
            
            dx = x_coords - mean[0]
            dy = y_coords - mean[1]
            power = -0.5 * (conic[0] * dx**2 + conic[2] * dy**2) - conic[1] * dx * dy
            alpha = np.minimum(0.99, opacity * np.exp(power))
            
            alpha_mask = alpha > 1/255.0
            if not np.any(alpha_mask):
                continue
            
            T = transmittance[min_y:max_y, min_x:max_x][alpha_mask]
            contribution = (alpha[alpha_mask] * T)[:, np.newaxis] * color
            out_image[min_y:max_y, min_x:max_x][alpha_mask] += contribution[:, np.newaxis] if contribution.ndim == 1 else contribution
            transmittance[min_y:max_y, min_x:max_x][alpha_mask] *= (1.0 - alpha[alpha_mask])

        return out_image


if __name__ == "__main__":
    # --- 1. 设置场景 ---
    num_gaussians = 4000
    print(f"创建 {num_gaussians} 个高斯点来组成一个甜甜圈...")

    R = 2.5
    r = 1.0
    theta = np.random.uniform(0, 2 * np.pi, num_gaussians)
    phi = np.random.uniform(0, 2 * np.pi, num_gaussians)

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    means3D = np.vstack([x, y, z]).T
    
    opacities = np.random.uniform(0.8, 0.95, (num_gaussians, 1))
    scales = np.random.uniform(0.05, 0.15, (num_gaussians, 3))
    rotations = np.random.rand(num_gaussians, 4) - 0.5
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    hue = theta / (2 * np.pi)
    colors_base = np.array([plt.cm.hsv(h)[:3] for h in hue])
    sh_degree = 1
    sh_coeffs_count = (sh_degree + 1)**2
    shs = np.random.rand(num_gaussians, sh_coeffs_count, 3) * 0.2 - 0.1
    shs[:, 0, :] = colors_base

    # --- 2. 准备渲染器和多角度视图 ---
    IMAGE_WIDTH = 400  # 可以减小分辨率以加快多视图的渲染速度
    IMAGE_HEIGHT = 400
    
    rasterizer = GaussianRasterizer(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    
    num_views = 4  # 我们想要渲染的视角数量
    fig, axs = plt.subplots(2, 2, figsize=(10, 10)) # 创建一个2x2的子图网格
    axs = axs.flatten() # 将2x2的轴数组展平为一维，方便循环

    camera_radius = 8.0 # 相机环绕物体的半径
    camera_height = 4.0 # 相机的高度

    for i in range(num_views):
        view_angle = i * (2 * np.pi / num_views) # 计算当前视角的角度
        print(f"\n--- 正在渲染视角 {i+1}/{num_views} (角度: {int(view_angle * 180 / np.pi)} 度) ---")

        # --- 计算当前视角的相机位置和矩阵 ---
        cam_pos = np.array([
            camera_radius * np.cos(view_angle),
            camera_radius * np.sin(view_angle),
            camera_height
        ])

        # 始终让相机朝向原点 (0,0,0)
        forward_dir = -cam_pos / np.linalg.norm(cam_pos)
        up_dir = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(forward_dir, up_dir)) > 0.99:
            up_dir = np.array([0.0, 1.0, 0.0])
        
        right_dir = np.cross(forward_dir, up_dir)
        right_dir /= np.linalg.norm(right_dir)
        cam_up_dir = np.cross(right_dir, forward_dir)
        
        rotation_matrix = np.eye(3)
        rotation_matrix[0, :] = right_dir
        rotation_matrix[1, :] = cam_up_dir
        rotation_matrix[2, :] = -forward_dir

        fov_rad = 50 * (math.pi / 180)
        tan_fovx = math.tan(fov_rad * 0.5)
        tan_fovy = math.tan(fov_rad * 0.5 * (IMAGE_HEIGHT / IMAGE_WIDTH))

        view_matrix = get_view_matrix(cam_pos, rotation_matrix.T)
        projection_matrix = get_projection_matrix(0.1, 100, fov_rad, fov_rad)

        # --- 3. 渲染当前视角 ---
        output_image = rasterizer.forward(
            means3D=means3D, opacities=opacities, scales=scales, rotations=rotations,
            shs=shs, view_matrix=view_matrix, proj_matrix=projection_matrix,
            cam_pos=cam_pos, tan_fovx=tan_fovx, tan_fovy=tan_fovy,
            bg_color=np.array([0.1, 0.1, 0.2])
        )

        # --- 4. 在子图中显示结果 ---
        ax = axs[i]
        ax.imshow(output_image)
        ax.set_title(f"View {i+1} (Angle: {int(view_angle * 180 / np.pi)}°)")
        ax.axis('off')

    # --- 5. 显示整个图像网格 ---
    fig.suptitle("Multi-View Rendering of 3D Torus", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    plt.show()