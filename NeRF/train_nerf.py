import os
import torch
import numpy as np
import imageio
from tqdm import tqdm, trange

# 导入我们重构后的模块
from nerf_model import NeRF, get_embedder
from nerf_utils import *
# 从单独的配置文件中导入所有参数
from config import config

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def train():
    """
    主训练和评估函数。
    """
    # ==========================================================================================
    # 1. 加载数据
    # ==========================================================================================
    images, poses, render_poses, hwf, i_split = load_blender_data(config['datadir'], config['half_res'], config['testskip'])
    print('Loaded blender', images.shape, render_poses.shape, hwf, config['datadir'])
    i_train, i_val, i_test = i_split
    
    near = 2.
    far = 6.

    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    if config['render_test']:
        render_poses = np.array(poses[i_test])

    basedir = config['basedir']
    expname = config['expname']
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # ==========================================================================================
    # 2. 创建模型和优化器
    # ==========================================================================================
    # [修改] create_nerf现在还会返回一个学习率调度器
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, scheduler = create_nerf(config)
    global_step = start

    bds_dict = {'near' : near, 'far' : far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    render_poses = torch.Tensor(render_poses).to(device)

    # ==========================================================================================
    # 3. [可选] 仅渲染模式
    # ==========================================================================================
    if config['render_only']:
        print('RENDER ONLY')
        with torch.no_grad():
            if config['render_test']:
                images = images[i_test]
            else:
                images = None
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if config['render_test'] else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, _ = render_path(render_poses, hwf, K, config['chunk'], render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=config['render_factor'])
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return

    # ==========================================================================================
    # 4. 准备训练数据
    # ==========================================================================================
    N_rand = config['N_rand']
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    print('Begin training')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    # ==========================================================================================
    # 5. 主训练循环
    # ==========================================================================================
    start = start + 1
    for i in trange(start, config['N_iters']):
        # --- 核心采样流程 ---
        img_i = np.random.choice(i_train)
        target = images[img_i]
        pose = poses[img_i, :3,:4]
        rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)
        coords = torch.reshape(coords, [-1,2])
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]

        # --- 核心优化步骤 ---
        rgb, disp, acc, extras = render(H, W, K, chunk=config['chunk'], rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # --- 更新学习率 ---
        # [修改] 使用scheduler.step()来更新学习率，代替手动计算
        scheduler.step()

        # --- 日志记录、保存和评估 ---
        if i % config['i_print'] == 0:
            # [修改] 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} LR: {current_lr:.6f}")
        
        if i % config['i_weights'] == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            # [修改] 保存scheduler的状态
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % config['i_video'] == 0 and i > 0:
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, config['chunk'], render_kwargs_test)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % config['i_testset'] == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, config['chunk'], render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        global_step += 1


def create_nerf(config):
    """
    实例化NeRF模型。
    """
    embed_fn, input_ch = get_embedder(config['multires'], config['i_embed'])

    input_ch_views = 0
    embeddirs_fn = None
    if config['use_viewdirs']:
        embeddirs_fn, input_ch_views = get_embedder(config['multires_views'], config['i_embed'])
    
    output_ch = 5 if config['N_importance'] > 0 else 4
    skips = [4]
    model = NeRF(D=config['netdepth'], W=config['netwidth'],
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=config['use_viewdirs']).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if config['N_importance'] > 0:
        model_fine = NeRF(D=config['netdepth_fine'], W=config['netwidth_fine'],
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=config['use_viewdirs']).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=config['netchunk'])

    optimizer = torch.optim.Adam(params=grad_vars, lr=config['lrate'], betas=(0.9, 0.999))
    
    # [新增] 创建学习率调度器
    # 计算gamma值以匹配原始的衰减逻辑
    decay_steps = config['lrate_decay'] * 1000
    decay_rate = 0.1
    gamma = decay_rate ** (1 / decay_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    start = 0
    basedir = config['basedir']
    expname = config['expname']

    if config['ft_path'] is not None and config['ft_path']!='None':
        ckpts = [config['ft_path']]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not config['no_reload']:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # [修改] 加载scheduler的状态
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : config['perturb'],
        'N_importance' : config['N_importance'],
        'network_fine' : model_fine,
        'N_samples' : config['N_samples'],
        'network_fn' : model,
        'use_viewdirs' : config['use_viewdirs'],
        'white_bkgd' : config['white_bkgd'],
        'raw_noise_std' : config['raw_noise_std'],
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # [修改] 返回scheduler
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, scheduler


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
