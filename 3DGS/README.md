### 一个纯 Python/NumPy 实现的 3DGS

官方代码链接：[graphdeco-inria/gaussian-splatting: Original reference implementation of "3D Gaussian Splatting for Real-Time Radiance Field Rendering"](https://github.com/graphdeco-inria/gaussian-splatting)

其他参考代码：https://github.com/SY-007-Research/3dgs_render_python

#### 核心功能

- **完整的3D高斯前向渲染管线**: 实现了从3D高斯数据到2D图像的完整渲染流程。
- **坐标变换与投影**:
  - 支持视图矩阵（View Matrix）和透视投影矩阵（Projection Matrix）的计算。
  - 提供将3D点从世界空间变换到裁剪空间的功能。
- **高斯模型计算**:
  - 支持将四元数（Quaternion）转换为旋转矩阵。
  - 实现了将3D协方差矩阵投影为2D协方差矩阵（圆锥曲线）的计算。
  - 支持从球谐函数（Spherical Harmonics）系数和视角方向计算最终颜色。
- **渲染优化**:
  - 包含视锥体裁剪（View Frustum Culling），只处理视野内的点。
  - 渲染前对高斯点按深度进行从后往前的排序，以保证正确的 Alpha 混合。
- **示例场景**:
  - 代码包含一个示例，程序化生成一个由数千个高斯点组成的甜甜圈（Torus）。
  - 从多个不同机位渲染该场景，并使用 `matplotlib` 展示结果。

#### 文件结构

- `rasterizer.py`: 项目主文件。包含了 `GaussianRasterizer` 类，负责整个渲染流程的编排和执行。同时，它也是运行示例场景的入口。
- `utils.py`: 辅助函数模块。包含了所有核心的数学和变换函数，如矩阵运算、四元数转换、协方差和球谐函数计算等。

#### 安装所需的库

```
pip install numpy matplotlib tqdm
```

#### 运行代码

```
python rasterizer.py
```

#### 期望输出

<img width="1129" height="846" alt="image" src="https://github.com/user-attachments/assets/a1fb9b2b-07b5-45b9-aec3-65a91f085879" />


