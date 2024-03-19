# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
from typing import List, Dict, Any
import os
import sys
import logging
logger = logging.getLogger('build-global')


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from utils.model_update_utils import (get_model_params,
                                      compute_visible_point_mask,
                                      rendering,
                                      sample_cameras,
                                      get_cameras_from_metadata,
                                      knn_filtering,
                                      faiss_knn,
                                      reset_opacity,
                                      make_index)
from utils.loss_utils import l1_loss, ssim


def distillation(global_model: GaussianModel,
                 local_model: GaussianModel,
                 global_metadatas: List[Dict[str, Any]],
                 img_height: List[int],
                 img_width: List[int],
                 fovx: List[float],
                 fovy: List[float],
                 viewmats: torch.Tensor,
                 bg_color: torch.Tensor,
                 lr_opacity: float,
                 lr_mlp: float,
                 wd_mlp: float,
                 lr_hash: float,
                 lr_avec: float,
                 resolution_scale: int,
                 n_epoch: int,
                 max_opacity: float=0.05,
                 far: int=100):
    target_images = []
    with torch.no_grad():
        # rendering target images from local model
        for i in range(len(viewmats)):
            rgb_l = rendering(local_model, img_height[i], img_width[i], fovx[i], fovy[i], viewmats[i], bg_color)[0]
            target_images.append(rgb_l)
        # rendering target images from global model
        target_camera_indices = sample_cameras(local_model, global_metadatas, max_cameras=len(target_images), far=far)
        g_h, g_w, g_fovx, g_fovy, g_vmats = get_cameras_from_metadata(global_metadatas, target_camera_indices)
        g_h = list(map(lambda x: x //resolution_scale, g_h))
        g_w = list(map(lambda x: x //resolution_scale, g_w))
        for i in range(len(g_vmats)):
            rgb_g = rendering(global_model, g_h[i], g_w[i], g_fovx[i], g_fovy[i], g_vmats[i], bg_color)[0]
            target_images.append(rgb_g)
        # add global model's views
        img_height += g_h
        img_width += g_w
        fovx += g_fovx
        fovy += g_fovy
        viewmats = torch.cat([viewmats, g_vmats])

        xyz_g, rot_g, scale_g, opacity_g, sh_g = get_model_params(global_model, preact=True, device='cpu')
        xyz_l, rot_l, scale_l, opacity_l, sh_l = get_model_params(local_model, preact=True, device='cpu')
        # reset opacity
        xyz_l_np = xyz_l.cpu().numpy()
        index = make_index(xyz_l_np)
        D, _ = faiss_knn(index, xyz_l_np, 2)
        eps = float(np.median(D[:, 1]))
        mask = knn_filtering(index, xyz_g.cpu().numpy(), eps)
        opacity_g[mask] = reset_opacity(opacity_g[mask], global_model.opacity_activation, global_model.inverse_opacity_activation, max_opacity)
        opacity_l = reset_opacity(opacity_l, local_model.opacity_activation, local_model.inverse_opacity_activation, max_opacity)
        # merge local and global model by concatenation
        new_params = dict(xyz=torch.cat([xyz_g, xyz_l]),
                          rotation=torch.cat([rot_g, rot_l]),
                          scaling=torch.cat([scale_g, scale_l]),
                          features_dc=torch.cat([sh_g[:, :1], sh_l[:, :1]]),
                          features_rest=torch.cat([sh_g[:, 1:], sh_l[:, 1:]]),
                          opacity=torch.cat([opacity_g, opacity_l]))
    global_model.set_params(new_params)

    app_vec = nn.Parameter(torch.zeros(len(viewmats), 32, device='cuda'))
    optimizer = optim.Adam([{'params': [global_model._opacity], 'lr': lr_opacity},
                            {'params': global_model.mlp.parameters(), 'lr': lr_mlp, 'weight_decay': wd_mlp},
                            {'params': global_model.pos_emb.parameters(), 'lr': lr_hash},
                            {'params': [app_vec], 'lr': lr_avec}],
                           lr=0.0, eps=1e-12, weight_decay=0.0)
    logger.info('update global model')
    index_list = list(range(len(viewmats)))
    for i in tqdm(range(n_epoch * len(viewmats))):
        if len(index_list) == 0:
            index_list = list(range(len(viewmats)))
        idx = index_list.pop(np.random.randint(0, len(index_list)))
        xyz = global_model.get_xyz.detach()
        glo_sh = global_model.mlp(dict(pos=global_model.pos_emb(xyz),
                              appearance=app_vec[idx])).reshape(len(xyz), -1, 3)
        rend_rgb = rendering(global_model,
                             img_height[idx],
                             img_width[idx],
                             fovx[idx],
                             fovy[idx],
                             viewmats[idx],
                             bg_color,
                             glo_sh)[0]
        loss = 0.8 * l1_loss(rend_rgb, target_images[idx]) + 0.2 * (1.0 - ssim(rend_rgb, target_images[idx]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # entropy minimization after updateing model for one epoch
        if i > len(viewmats):
            grad = global_model._opacity.grad
            opacity = global_model.get_opacity[grad!=0]
            reg = (- opacity * torch.log(opacity.clamp(min=1e-8))
                   - (1 - opacity) * torch.log((1 - opacity).clamp(min=1e-8))).mean() * 1e-2
            reg.backward()
        optimizer.step()

    return global_model


def update_model(global_params: Dict[str, Any],
                 client_model: GaussianModel,
                 client_metadatas: List[Dict[str, Any]],
                 global_model_camera_meta: List[Dict[str, Any]],
                 min_opacity: float,
                 lr_opacity: float,
                 lr_mlp: float,
                 wd_mlp: float,
                 lr_hash: float,
                 lr_avec: float,
                 n_epoch: int,
                 bg_color: torch.Tensor,
                 resolution_scale: int=1,
                 far: int=100):
    # get camera intrinsic
    image_height, image_width, fovx, fovy, viewmats = get_cameras_from_metadata(client_metadatas)
    image_height = list(map(lambda x: x //resolution_scale, image_height))
    image_width = list(map(lambda x: x //resolution_scale, image_width))
    # get visible Gaussians
    vis_msk = compute_visible_point_mask(global_params['xyz'], client_metadatas, 'cpu')
    logger.info(f"#global model's points: {len(global_params['xyz'])} ({vis_msk.sum()} visible points)")
    logger.info(f"#local model's points: {len(client_model._xyz.data)}")
    xyz_g = global_params['xyz']
    rot_g = global_params['rotation']
    scale_g = global_params['scaling']
    opacity_g = global_params['opacity']
    sh_g = torch.cat([global_params['features_dc'], global_params['features_rest']], 1)
    vis_xyz_g = xyz_g[vis_msk]
    vis_rot_g = rot_g[vis_msk]
    vis_scale_g = scale_g[vis_msk]
    vis_opacity_g = opacity_g[vis_msk]
    vis_sh_g = sh_g[vis_msk]
    tmp_global_model = GaussianModel(client_model.max_sh_degree)
    logger.info(f'#points before model update: {len(vis_xyz_g)}')
    new_params = dict(xyz=vis_xyz_g,
                      rotation=vis_rot_g,
                      scaling=vis_scale_g,
                      features_dc=vis_sh_g[:, :1],
                      features_rest=vis_sh_g[:, 1:],
                      opacity=vis_opacity_g,
                      app_mlp=global_params['app_mlp'],
                      app_pos_emb=global_params['app_pos_emb'])
    tmp_global_model.set_params(new_params)

    tmp_global_model = distillation(tmp_global_model,
                                    client_model,
                                    global_model_camera_meta,
                                    image_height,
                                    image_width,
                                    fovx,
                                    fovy,
                                    viewmats,
                                    bg_color,
                                    lr_opacity,
                                    lr_mlp,
                                    wd_mlp,
                                    lr_hash,
                                    lr_avec,
                                    resolution_scale,
                                    n_epoch,
                                    min_opacity,
                                    far=far)

    vis_xyz_g, vis_rot_g, vis_scale_g, vis_opacity_g, vis_sh_g = get_model_params(tmp_global_model, preact=True, device='cpu')
    app_mlp = tmp_global_model.mlp.state_dict()
    app_pos_emb = tmp_global_model.pos_emb.state_dict()
    # prune points
    prune_mask = (vis_opacity_g.sigmoid() > min_opacity).reshape(-1)
    if prune_mask.any():
        logger.info(f'prune {(~prune_mask).sum()} points')
        vis_xyz_g = vis_xyz_g[prune_mask]
        vis_rot_g = vis_rot_g[prune_mask]
        vis_scale_g = vis_scale_g[prune_mask]
        vis_opacity_g = vis_opacity_g[prune_mask]
        vis_sh_g = vis_sh_g[prune_mask]
    xyz_g = torch.cat([xyz_g[~vis_msk], vis_xyz_g])
    rot_g = torch.cat([rot_g[~vis_msk], vis_rot_g])
    scale_g = torch.cat([scale_g[~vis_msk], vis_scale_g])
    opacity_g = torch.cat([opacity_g[~vis_msk], vis_opacity_g])
    sh_g = torch.cat([sh_g[~vis_msk], vis_sh_g])

    new_params = dict(xyz=xyz_g,
                      rotation=rot_g,
                      scaling=scale_g,
                      features_dc=sh_g[:, :1],
                      features_rest=sh_g[:, 1:],
                      opacity=opacity_g,
                      app_mlp=app_mlp,
                      app_pos_emb=app_pos_emb)
    logger.info(f'#points after model update: {len(xyz_g)}')
    return new_params


def _update_model(global_params, client_model_index, metadatas, client_metadatas, global_model_cam_list, intersection, bg_color, load_iter, args):
    # load local model
    client_model_file = os.path.join(args.model_dir,
                                     client_model_index,
                                     'point_cloud/iteration_' + str(load_iter) + '/point_cloud.ply')
    client_model = GaussianModel(args.sh_degree)
    client_model.load_ply(client_model_file)
    logger.info(f'update model with {client_model_index}-th clients')
    g_sub_l = np.setdiff1d(global_model_cam_list, intersection)
    global_model_camera_meta = [metadatas[fname.split('.')[0]] for fname in g_sub_l]
    global_params = update_model(global_params, client_model, client_metadatas,
                                 global_model_camera_meta, args.min_opacity, args.lr_opacity,
                                 args.lr_mlp, args.wd_mlp, args.lr_hash, args.lr_avec,
                                 args.n_kd_epoch, bg_color, args.resolution, far=args.far)
    return global_params


def check_buffer(global_params,
                 client_buffer,
                 metadatas,
                 load_iter,
                 bg_color,
                 global_model_cam_list,
                 n_added_client,
                 args):
    tmp_client_buffer = []
    for b_client_idx, b_client_cam_list in client_buffer:
        torch.cuda.empty_cache()
        intersection = np.intersect1d(global_model_cam_list, b_client_cam_list)
        # client selection
        if len(intersection) < args.overlap_img_threshold:
            tmp_client_buffer.append([b_client_idx, b_client_cam_list])
            continue
        client_model_index = b_client_idx.split('.')[0]
        client_metadatas = [metadatas[fname.split('.')[0]] for fname in b_client_cam_list]
        global_params = _update_model(global_params, client_model_index, metadatas, client_metadatas,
                                      global_model_cam_list, intersection, bg_color, load_iter, args)
        # update global model's camera list
        global_model_cam_list = np.union1d(global_model_cam_list, b_client_cam_list)
        n_added_client += 1
        # save model
        if (n_added_client % args.save_freq) == 0:
            torch.save(global_params, os.path.join(args.output_dir, f'global_model_{n_added_client}clients.pth'))
    updated = len(tmp_client_buffer) < len(client_buffer)
    return global_params, tmp_client_buffer, global_model_cam_list, updated, n_added_client


def main(args):
    metadata_dir = os.path.join(args.dataset_dir, 'train/metadata')
    logger.info('load metadata')
    # load metadata including camera intrinsic and extrinsic
    metadata_files = sorted(os.listdir(metadata_dir))
    metadatas = {}
    for fname in tqdm(metadata_files):
        file_idx = fname.split('.')[0]
        metadatas[file_idx] = torch.load(os.path.join(metadata_dir, fname))
    # load image indices in clients data
    logger.info('load image lists')
    index_files = sorted(os.listdir(args.index_dir))
    if args.shuffle:
        index_files = list(np.random.permutation(index_files))
    if args.n_clients > 0:
        index_files = index_files[:args.n_clients]
    image_lists = [list(np.loadtxt(os.path.join(args.index_dir, fname), dtype=str))
                   for fname in index_files if '.txt' in fname]
    # load a 0-th local model as a global model
    logger.info('initialize global model')
    seed_model_index = index_files.pop(0).split('.')[0]
    load_iter = args.load_iteration
    seed_model_file = os.path.join(args.model_dir,
                                   seed_model_index,
                                   'point_cloud/iteration_' + str(load_iter) + '/point_cloud.ply')
    global_model = GaussianModel(args.sh_degree)
    global_model.load_ply(seed_model_file)
    # get model params
    xyz_g, rot_g, scale_g, opacity_g, sh_g = get_model_params(global_model, preact=True, device='cpu')
    global_params = dict(xyz=xyz_g,
                        rotation=rot_g,
                        scaling=scale_g,
                        features_dc=sh_g[:, :1],
                        features_rest=sh_g[:, 1:],
                        opacity=opacity_g,
                        app_mlp=global_model.mlp.state_dict(),
                        app_pos_emb=global_model.pos_emb.state_dict())
    del global_model
    # global model's camera list
    global_model_cam_list = image_lists.pop(0)
    # placeholder
    client_buffer = []
    # set background color
    bg_color = torch.Tensor([1., 1., 1.]).cuda() if args.white_bg else torch.Tensor([0., 0., 0.]).cuda()
    n_added_client = 1
    for client_idx, client_cam_list in zip(index_files, image_lists):
        intersection = np.intersect1d(global_model_cam_list, client_cam_list)
        # client selection
        if len(intersection) < args.overlap_img_threshold:
            client_buffer.append([client_idx, client_cam_list])
            continue
        logger.info('---')
        # load a local model
        client_model_index = client_idx.split('.')[0]
        client_metadatas = [metadatas[fname.split('.')[0]] for fname in client_cam_list]
        global_params = _update_model(global_params, client_model_index, metadatas, client_metadatas,
                                      global_model_cam_list, intersection, bg_color, load_iter, args)
        # update global model's camera list
        global_model_cam_list = np.union1d(global_model_cam_list, client_cam_list)
        n_added_client += 1
        # save model
        if (n_added_client % args.save_freq) == 0:
            torch.save(global_params, os.path.join(args.output_dir, f'global_model_{n_added_client}clients.pth'))
        # aggregate buffered models
        while True:
            global_params, client_buffer, global_model_cam_list, updated, n_added_client = check_buffer(global_params, client_buffer, metadatas,
                                                                                                        load_iter, bg_color, global_model_cam_list,
                                                                                                        n_added_client, args)
            if not updated:
                break

    torch.save(global_params, os.path.join(args.output_dir, f'global_model.pth'))


if __name__=='__main__':
    import random
    import argparse
    parser = argparse.ArgumentParser()
    ### directory args
    parser.add_argument('--output-dir', '-o', default='./', type=str,
                        help='/path/to/output-dir')
    parser.add_argument('--model-dir', '-m', required=True, type=str,
                        help='/path/to/client-model-dir')
    parser.add_argument('--index-dir', '-i', required=True, type=str,
                        help='/path/to/image-list-file-dir')
    parser.add_argument('--dataset-dir', '-data', required=True, type=str,
                        help='/path/to/dataset-dir')
    ### experimental setup
    parser.add_argument('--shuffle', action='store_true',
                        help='if True, randomly aggregating client models')
    ### model args
    parser.add_argument('--sh-degree', default=2, type=int)
    parser.add_argument('--load-iteration', '-liter', default='20000', type=str)
    parser.add_argument('--white-bg', '-w', action='store_true')
    ### alignment args
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate for alignment')
    parser.add_argument('--overlap-img-threshold', '-oth', default=20, type=int)
    ### aggregation args
    parser.add_argument('--min-opacity', '-min-o', default=0.005, type=float)
    parser.add_argument('--n-clients', default=-1, type=int)
    parser.add_argument('--n-kd-epoch', default=5, type=int)
    ### optimizer args
    parser.add_argument('--lr-opacity', '-lro', default=0.05, type=float)
    parser.add_argument('--lr-mlp', '-lrm', default=1e-4, type=float)
    parser.add_argument('--wd-mlp', default=1e-4, type=float)
    parser.add_argument('--lr-hash', '-lrh', default=1e-4, type=float)
    parser.add_argument('--lr-avec', default=1e-3, type=float)
    ### misc
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--save-freq', default=100, type=int)
    parser.add_argument('--resolution', '-r', default=4, type=int)
    parser.add_argument('--far', default=100, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    main(args)
