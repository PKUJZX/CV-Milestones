官方代码链接：[bmild/nerf: Code release for NeRF (Neural Radiance Fields)](https://github.com/bmild/nerf)

其他参考代码：https://github.com/yenchenlin/nerf-pytorch

#### 第一步：准备环境和依赖库

```
pip install torch numpy imageio imageio-ffmpeg opencv-python tqdm
```

#### 第二步：下载和放置数据集

**下载数据**:  http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip

**解压并放置数据**: 下载 `nerf_synthetic.zip` 并将其解压到 `data` 文件夹内。

```
config.py
nerf_model.py
nerf_utils.py
train_nerf.py
data/
└── nerf_synthetic/
    ├── lego/
```

#### 第三步：开始训练模型

```
python train_nerf.py
```

你会在终端看到一个进度条，显示训练的迭代次数、损失（Loss）和图像质量指标（PSNR）。模型权重会定期保存在 `./logs/lego_exp_final/` 目录下。

#### 第四步：使用训练好的模型进行渲染

**修改配置文件**: 打开 `config.py` 文件，将 `render_only` 设置为 `True`。

**重新运行脚本**: 再次运行 `train_nerf.py`。这次它不会进行训练，而是会加载最新的模型权重，并执行渲染任务。

```
python train_nerf.py
```


渲染完成后，生成的视频（如 `video.mp4`）或图像序列会保存在 `logs` 目录下一个新的子文件夹中，例如 `renderonly_path_199999`。

<img width="1668" height="746" alt="image" src="https://github.com/user-attachments/assets/b8e01a2b-6bc1-43cb-a5b5-8144403451b9" />
