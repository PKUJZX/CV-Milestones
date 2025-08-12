import torch
from torchdiffeq import odeint

# --- 辅助函数 ---

def mean_flat(x):
    """
    计算张量除了第一个维度（通常是 batch size）外所有维度的平均值。
    常用于计算损失。
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def expand_t_like_x(t, x):
    """
    将时间张量 t 的形状扩展到与数据张量 x 匹配，以便进行广播操作。
    例如, t: [B] -> [B, C, H, W]
    """
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


# --- 路径定义 ---

class LinearPath:
    """
    定义从噪声 x0 到数据 x1 的线性插值路径。
    xt = (1-t) * x0 + t * x1
    """
    def _compute_coeffs(self, t):
        """计算线性路径的系数。"""
        alpha_t = t          # 数据 x1 的系数
        d_alpha_t = torch.ones_like(t) # alpha_t 对 t 的导数
        sigma_t = 1 - t      # 噪声 x0 的系数
        d_sigma_t = -torch.ones_like(t) # sigma_t 对 t 的导数
        return alpha_t, d_alpha_t, sigma_t, d_sigma_t

    def plan(self, t, x0, x1):
        """
        根据时间 t 计算路径上的点 xt 和该点的真实速度 ut。
        :param t: 时间步张量
        :param x0: 初始噪声
        :param x1: 目标数据
        :return: (时间t, 路径上的点xt, 真实速度ut)
        """
        t_expanded = expand_t_like_x(t, x1)
        alpha_t, d_alpha_t, sigma_t, d_sigma_t = self._compute_coeffs(t_expanded)
        
        # 计算路径上的插值点 xt
        xt = alpha_t * x1 + sigma_t * x0
        # 计算该点的真实速度 ut (xt 对 t 的导数)
        ut = d_alpha_t * x1 + d_sigma_t * x0
        
        return t, xt, ut


# --- ODE 求解器 ---

class ODESolver:
    """
    使用 torchdiffeq 库来求解常微分方程 (ODE) 的包装器。
    用于在采样阶段从噪声生成数据。
    """
    def __init__(self, drift_fn, t0=0.0, t1=1.0, num_steps=50, atol=1e-6, rtol=1e-3):
        self.drift_fn = drift_fn  # ODE 的漂移函数，通常是训练好的模型
        self.t = torch.linspace(t0, t1, num_steps) # 积分的时间点
        self.atol = atol # 绝对容忍度
        self.rtol = rtol # 相对容忍度

    def sample(self, x, model, **model_kwargs):
        """
        从初始点 x (通常是噪声) 开始，通过求解 ODE 生成样本。
        """
        device = x.device
        
        # 定义 ODE 函数: dx/dt = drift_fn(x, t)
        def _ode_func(t, x_t):
            t_tensor = torch.ones(x_t.size(0), device=device) * t
            return self.drift_fn(x_t, t_tensor, model, **model_kwargs)

        # 使用 'dopri5' 方法求解 ODE
        solution = odeint(
            _ode_func,
            x,
            self.t.to(device),
            method='dopri5',
            atol=self.atol,
            rtol=self.rtol
        )
        return solution


# --- 核心传输框架 ---

class Transport:
    """
    管理训练过程的核心类。
    它负责采样时间、构建路径和计算训练损失。
    """
    def __init__(self, train_eps=1e-5, sample_eps=1e-3):
        self.path_sampler = LinearPath()
        self.train_eps = train_eps  # 训练时避免 t=0 或 t=1 的小量
        self.sample_eps = sample_eps # 采样时避免 t=0 或 t=1 的小量

    def _sample_time_and_noise(self, x1):
        """为一批数据 x1 采样随机时间和初始噪声 x0。"""
        t0, t1 = self.train_eps, 1.0 - self.train_eps
        t = torch.rand((x1.shape[0],), device=x1.device) * (t1 - t0) + t0
        x0 = torch.randn_like(x1)
        return t, x0
        
    def training_losses(self, model, x1, model_kwargs=None):
        """
        计算训练损失。
        核心思想是让模型预测的速度逼近路径上的真实速度。
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # 1. 采样时间和噪声
        t, x0 = self._sample_time_and_noise(x1)
        # 2. 计算路径上的点 xt 和真实速度 true_velocity
        _, xt, true_velocity = self.path_sampler.plan(t, x0, x1)
        
        # 3. 让模型预测在 xt, t 点的速度
        predicted_velocity = model(xt, t, **model_kwargs)
        
        # 4. 计算预测速度和真实速度之间的均方误差损失
        loss = mean_flat((predicted_velocity - true_velocity) ** 2)
        
        return {'loss': loss}

    def _get_drift_function(self):
        """
        获取 ODE 的漂移函数。在我们的框架中，漂移函数就是训练好的模型本身。
        """
        return lambda x, t, model, **kwargs: model(x, t, **kwargs)


# --- 采样器 ---

class Sampler:
    """
    管理采样过程的核心类。
    它将 Transport 框架和 ODESolver 连接起来。
    """
    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift = self.transport._get_drift_function()

    def get_sampler_fn(self, num_steps=50, atol=1e-6, rtol=1e-3):
        """
        配置并返回一个 ODE 求解器函数，用于从噪声生成样本。
        """
        t0 = self.transport.sample_eps
        t1 = 1.0 - self.transport.sample_eps
        
        # 创建一个 ODESolver 实例
        solver = ODESolver(
            drift_fn=self.drift,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol
        )
        
        return solver.sample
