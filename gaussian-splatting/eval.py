# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
import os
import sys
import json
import logging
logger = logging.getLogger('eval')

from PIL import Image

import cv2
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.model_update_utils import (meganerf2colmap,
                                      rendering,
                                      get_model_params)


def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
    to_use = scalar_tensor.view(-1)
    while to_use.shape[0] > 2 ** 24:
        to_use = to_use[::2]

    mi = torch.quantile(to_use, 0.05)
    ma = torch.quantile(to_use, 0.95)

    scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    scalar_tensor = scalar_tensor.clamp_(0, 1)

    scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
    return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def evaluation(global_params,
               dataset_dir,
               val_image_lists,
               val_metadatas,
               bg_color,
               sh_degree,
               n_iter,
               lr,
               resolution_scale=1):
    rendered_images = []
    rendered_depths = []
    psnrs = []
    ssims = []
    lpipss = []
    global_model = GaussianModel(sh_degree)
    logger.info('evaluate')
    for img_fname, meta in zip(val_image_lists, val_metadatas):
        # load data
        viewmat = meganerf2colmap(meta['c2w']).cuda()
        image_height = meta['H']
        image_width = meta['W']
        fx, fy, _, _ = meta['intrinsics']
        if isinstance(fx, torch.Tensor):
            fx = fx.item()
            fy = fy.item()
        fovx = focal2fov(fx, image_width)
        fovy = focal2fov(fy, image_height)
        image_width //= resolution_scale
        image_height //= resolution_scale
        global_model.set_params(global_params)
        image_PIL = Image.open(os.path.join(dataset_dir, 'val/rgbs', img_fname))
        image_PIL = image_PIL.resize((image_width, image_height))
        image = torch.from_numpy(np.array(image_PIL)) / 255.0
        image = image.permute(2, 0, 1)[:3].cuda()
        # optimizer appearance vector
        app_vec = torch.nn.Parameter(torch.zeros(32).cuda())
        optimizer = optim.Adam([app_vec], lr=lr, eps=1e-12)
        with torch.no_grad():
            pos_emb = global_model.pos_emb(global_params['xyz'].cuda())
        init_loss = None
        for _ in range(n_iter):
            glo_sh = global_model.mlp(dict(pos=pos_emb, appearance=app_vec)).reshape(len(pos_emb), -1, 3)
            rend_image = rendering(global_model, image_height, image_width, fovx, fovy, viewmat, bg_color, glo_sh)[0]
            ## remove right-side pixels
            loss = (rend_image[None, ..., :rend_image.shape[-1]//2] - image[None, ..., :image.shape[-1]//2]).square().mean()
            if init_loss is None:
                init_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if init_loss is not None:
            logger.info(f'loss : {init_loss} -> {loss.item()}')
        # compute metrics
        with torch.no_grad():
            glo_sh = global_model.mlp(dict(pos=pos_emb, appearance=app_vec)).reshape(len(pos_emb), -1, 3)
            rend_image = rendering(global_model, image_height, image_width, fovx, fovy, viewmat, bg_color, glo_sh)[0]
            rendered_images.append(rend_image.cpu())
            depth = rendering(global_model, image_height, image_width, fovx, fovy, viewmat, bg_color, depth=True)[0]
            depth = visualize_scalars(torch.log(depth + 1e-8).detach().cpu())
            rendered_depths.append(depth)
            ## remove left-side pixels
            rend_image = rend_image[..., rend_image.shape[-1]//2:]
            image = image[..., image.shape[-1]//2:]
        psnrs.append(psnr(rend_image[None], image[None]).item())
        logger.info(f'{img_fname}')
        ssims.append(ssim(rend_image[None], image[None]).item())
        lpipss.append(lpips(rend_image[None], image[None], net_type='vgg').item())
        logger.info(f'PSNR: {psnrs[-1]}')
        logger.info(f'SSIM: {ssims[-1]}')
        logger.info(f'LPIPS: {lpipss[-1]}')
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ssim = torch.tensor(ssims).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()
    del global_model
    torch.cuda.empty_cache()
    logger.info('---')
    logger.info(f'AVG. PSNR: {avg_psnr}')
    logger.info(f'AVG. SSIM: {avg_ssim}')
    logger.info(f'AVG. LPIPS: {avg_lpips}')
    return rendered_images, rendered_depths, avg_psnr, avg_ssim, avg_lpips


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    ### directory args
    parser.add_argument('--output-dir', '-o', default='./', type=str,
                        help='/path/to/output-dir')
    parser.add_argument('--global-params', '-g', required=True, type=str,
                        help='/path/to/global-parameters')
    parser.add_argument('--dataset-dir', '-data', required=True, type=str,
                        help='/path/to/dataset-dir')
    ### appearance args
    parser.add_argument('--lr', default=5e-2, type=float,
                        help='learning rate')
    parser.add_argument('--n-iter', default=100, type=int)
    ### model args
    parser.add_argument('--sh-degree', default=2, type=int)
    parser.add_argument('--white-bg', '-w', action='store_true')
    ### misc
    parser.add_argument('--resolution', '-r', default=4, type=int)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # setup logger
    logger.setLevel(logging.INFO)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    plain_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S')
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.INFO)
    logger.addHandler(s_handler)
    f_handler = logging.FileHandler(os.path.join(args.output_dir, 'console.log'))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    logger.info(f'load global model from {args.global_params}')
    if '.ply' in args.global_params:
        tmp_model = GaussianModel(args.sh_degree)
        tmp_model.load_ply(args.global_params)
        global_params = get_model_params(tmp_model, True, device='cpu')
        global_params = {'xyz': global_params[0],
                         'rotation': global_params[1],
                         'scaling': global_params[2],
                         'opacity': global_params[3],
                         'features_dc': global_params[4][:, :1],
                         'features_rest': global_params[4][:, 1:],
                         'app_mlp': tmp_model.mlp.state_dict(),
                         'app_pos_emb': tmp_model.pos_emb.state_dict()}
        del tmp_model
    else:
        global_params = torch.load(args.global_params)
    logger.info(f'#Gaussians {len(global_params["xyz"])}')
    logger.info('load metadata')
    # set background color
    bg_color = torch.Tensor([1., 1., 1.]).cuda() if args.white_bg else torch.Tensor([0., 0.,0.]).cuda()
    # evaluation
    val_image_lists = sorted(os.listdir(os.path.join(args.dataset_dir, 'val/rgbs')))
    val_metadatas = [torch.load(os.path.join(args.dataset_dir, 'val/metadata', f.split('.')[0]+'.pt')) for f in val_image_lists]
    images, depths, psnr, ssim, lpips = evaluation(global_params,
                                                   args.dataset_dir,
                                                   val_image_lists,
                                                   val_metadatas,
                                                   bg_color,
                                                   args.sh_degree,
                                                   args.n_iter,
                                                   args.lr,
                                                   args.resolution)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(dict(psnr=psnr, ssim=ssim, lpips=lpips), f)

    for fname, img in zip(val_image_lists, images):
        save_image(img, os.path.join(args.output_dir, fname))

    for fname, img in zip(val_image_lists, depths):
        plt.imsave(os.path.join(args.output_dir, 'depth-' + fname), img)
