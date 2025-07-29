import math
import numpy as np

# --- 矩阵和变换函数 ---

def get_view_matrix(cam_pos=np.array([0, 0, 5]), rotation=np.eye(3)):
    """
    根据相机位置和旋转计算视图矩阵。
    Args:
        cam_pos (np.ndarray): 相机位置。
        rotation (np.ndarray): 相机旋转矩阵。
    Returns:
        np.ndarray: (4, 4) 视图矩阵。
    """
    # Z轴反转，因为OpenGL相机看向-Z
    R = rotation.T
    t = -R @ cam_pos
    
    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = t
    return view

def get_projection_matrix(znear, zfar, fovX, fovY):
    """
    计算透视投影矩阵。
    Args:
        znear (float): 近裁剪面距离。
        zfar (float): 远裁剪面距离。
        fovX (float): X方向的视场角（弧度）。
        fovY (float): Y方向的视场角（弧度）。
    Returns:
        np.ndarray: (4, 4) 投影矩阵。
    """
    tanHalfFovY = math.tan(fovY / 2.0)
    tanHalfFovX = math.tan(fovX / 2.0)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    
    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0
    return P

def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    使用4x4矩阵变换一批3D点。
    Args:
        points (np.ndarray): (N, 3) 3D点。
        matrix (np.ndarray): (4, 4) 变换矩阵。
    Returns:
        np.ndarray: (N, 4) 变换后的齐次坐标点。
    """
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    return points_hom @ matrix.T

def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """
    将一批四元数(w, x, y, z)转换为旋转矩阵。
    Args:
        quat (np.ndarray): (N, 4) 四元数数组。
    Returns:
        np.ndarray: (N, 3, 3) 旋转矩阵数组。
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    N = quat.shape[0]
    R = np.zeros((N, 3, 3))

    R[:, 0, 0] = 1 - 2*y*y - 2*z*z
    R[:, 0, 1] = 2*x*y - 2*z*w
    R[:, 0, 2] = 2*x*z + 2*y*w
    R[:, 1, 0] = 2*x*y + 2*z*w
    R[:, 1, 1] = 1 - 2*x*x - 2*z*z
    R[:, 1, 2] = 2*y*z - 2*x*w
    R[:, 2, 0] = 2*x*z - 2*y*w
    R[:, 2, 1] = 2*y*z + 2*x*w
    R[:, 2, 2] = 1 - 2*x*x - 2*y*y
    
    return R

# --- 高斯计算函数 ---

def compute_cov2d(p_view: np.ndarray, focal_x: float, focal_y: float, cov3D: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
    """
    计算投影到2D屏幕空间的协方差矩阵。
    """
    tz = p_view[:, 2, np.newaxis]
    
    J = np.zeros((p_view.shape[0], 2, 3))
    J[:, 0, 0] = focal_x / tz[:, 0]
    J[:, 0, 2] = -(focal_x * p_view[:, 0]) / (tz[:, 0]**2)
    J[:, 1, 1] = focal_y / tz[:, 0]
    J[:, 1, 2] = -(focal_y * p_view[:, 1]) / (tz[:, 0]**2)

    W = view_matrix[:3, :3].T
    T = J @ W
    
    cov2D = T @ cov3D @ T.transpose(0, 2, 1)

    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3
    
    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    inv_det = 1.0 / det
    
    conics = np.zeros((cov2D.shape[0], 3))
    conics[:, 0] = cov2D[:, 1, 1] * inv_det
    conics[:, 1] = -cov2D[:, 0, 1] * inv_det
    conics[:, 2] = cov2D[:, 0, 0] * inv_det
    
    return conics

def compute_color_from_sh(shs: np.ndarray, view_dirs: np.ndarray) -> np.ndarray:
    """
    从球谐函数系数计算颜色。
    """
    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2 = np.array([1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396])
    
    sh_dim = shs.shape[1]
    
    result = SH_C0 * shs[:, 0, :]
    
    if sh_dim > 1:
        x, y, z = view_dirs[:, 0], view_dirs[:, 1], view_dirs[:, 2]
        result = result - SH_C1 * y[:, None] * shs[:, 1, :] \
                        + SH_C1 * z[:, None] * shs[:, 2, :] \
                        - SH_C1 * x[:, None] * shs[:, 3, :]
    
    if sh_dim > 4:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        result = result + SH_C2[0] * xy[:, None] * shs[:, 4, :] \
                        + SH_C2[1] * yz[:, None] * shs[:, 5, :] \
                        + SH_C2[2] * (2.0*zz - xx - yy)[:, None] * shs[:, 6, :] \
                        + SH_C2[3] * xz[:, None] * shs[:, 7, :] \
                        + SH_C2[4] * (xx - yy)[:, None] * shs[:, 8, :]
    
    result += 0.5
    return np.clip(result, 0.0, 1.0)