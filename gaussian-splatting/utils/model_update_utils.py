# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
from typing import Optional, Tuple, List, Dict, Any

import math

import faiss
import torch
import numpy as np

from .graphics_utils import getProjectionMatrix, focal2fov

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


RDF_TO_DRB = torch.Tensor([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, -1]])


def reset_opacity(opacity, activation, inverse_activation, max_op=0.01):
    return inverse_activation(activation(opacity).clamp(max=max_op))


def faiss_knn(index, query, k):
    D, indices = index.search(query, k=k)
    return D, indices


def make_index(data):
    quantizer = faiss.IndexFlatL2(3)
    index = faiss.IndexIVFFlat(quantizer, 3, int(np.sqrt(len(data))), faiss.METRIC_L2)
    index.train(data)
    index.add(data)
    return index


def knn_filtering(index, query, eps):
    lims, _, _ = index.range_search(query, eps)
    return (lims[1:] - lims[:-1]) > 0


def compute_projection(fovx, fovy, extrinsic):
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fovx, fovY=fovy).transpose(0,1).to(extrinsic.device)
    full_proj_transform = (extrinsic.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    return full_proj_transform


@torch.no_grad()
def in_frustum_mask(points, viewmatrix, projmatrix, far: float=100.0, bound: float=1.3):
    """This function is based on `in_frustum` in auxiliary.h of diff-gaussian-rasterization
    
    Args:
        points (torch.Tensor): gaussian center that is a Tensor of shape (#points, 3)
        viewmatrix (torch.Tensor): world-to-camera matrix that is a column-major (=transposed) Tensor of shape (4, 4)
        projmatrix (torch.Tensor): projection matrix that is a column-major (=transposed) Tensor of shape (4, 4)

    Returns:
        masks (torch.Tensor): binary mask that is a Tensor of shape (#points,)
    """
    p_hom = projmatrix.T[None, :4, :3] @ points.reshape(-1, 3, 1) + projmatrix.T[None, :4, -1:]
    p_w = 1 / (p_hom[:, -1] + 1e-7)
    p_proj = p_hom[:, :3, 0] * p_w
    p_view = viewmatrix.T[None, :3, :3] @ points.reshape(-1, 3, 1) + viewmatrix.T[None, :3, -1:]
    
    return torch.logical_and(torch.logical_and(p_proj[:, 0].abs() <= bound, p_proj[:, 1].abs() <= bound),
                             torch.logical_and(p_view[:, -1, 0] > 0., p_view[:, -1, 0] < far))


@torch.no_grad()
def compute_visible_point_mask(xyz: torch.Tensor, metadatas: List[Dict[str, Any]], device='cuda'):
    """
    Returns:
        masks (torch.Tensor): visible point mask that is a bool Tensor of shape (#points,)
    """
    viewmat = torch.stack([meganerf2colmap(m['c2w']) for m in metadatas]).to(xyz.device)
    fx, fy, cx, cy = list(zip(*[m['intrinsics'] for m in metadatas]))
    Hs = [m['H'] for m in metadatas]
    Ws = [m['W'] for m in metadatas]
    fovxs = [focal2fov(f.item(), W) for f, W in zip(fx, Ws)]
    fovys = [focal2fov(f.item(), H) for f, H in zip(fy, Hs)]
    proj_transform = torch.stack([compute_projection(fovx, fovy, vm)
                                  for fovx, fovy, vm in zip(fovxs, fovys, viewmat)])
    return sum([in_frustum_mask(xyz, viewm, proj) for viewm, proj in zip(viewmat, proj_transform)]).bool()


def meganerf2colmap(c2w: torch.Tensor, return_w2c: bool=True, reorder: bool=True):
    if reorder:
        c2w = torch.cat([-c2w[:, 1:2], c2w[:, :1], c2w[:, 2:]], 1)
        # inverse transform of https://github.com/cmusatyalab/mega-nerf/blob/main/scripts/colmap_to_mega_nerf.py#L346-L349
        c2w = torch.cat([RDF_TO_DRB.inverse() @ c2w[:3, :3] @ RDF_TO_DRB,
                         RDF_TO_DRB.inverse() @ c2w[:3, 3:]], -1)
    # camera-to-world to world-to-camera
    if return_w2c:
        R = c2w[:3, :3].T
        t = - R @ c2w[:3, -1:]
        extrinsic = torch.cat([torch.cat([R, t], -1), torch.eye(4)[-1:]], 0).T # column-major
    else:
        extrinsic = torch.cat([c2w, torch.eye(4, device=c2w.device)[-1:]], 0)
    return extrinsic


def rendering(model, img_height, img_width, fovx, fovy, extrinsic, bg_color, sh_modifier=None, depth=False):
    screenspace_points = torch.zeros_like(model.get_xyz, dtype=model.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    proj_transform = compute_projection(fovx, fovy, extrinsic)
    raster_settings = GaussianRasterizationSettings(image_height=img_height,
                                                    image_width=img_width,
                                                    tanfovx=tanfovx,
                                                    tanfovy=tanfovy,
                                                    bg=bg_color,
                                                    scale_modifier=1.0,
                                                    sh_degree=model.max_sh_degree,
                                                    viewmatrix = extrinsic,
                                                    projmatrix = proj_transform,
                                                    campos = extrinsic[3, :3],
                                                    prefiltered=False,
                                                    debug=False)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = model.get_xyz
    means2D = screenspace_points
    opacity = model.get_opacity
    scales = model.get_scaling
    rotations = model.get_rotation
    cov3D_precomp = None
    colors_precomp = None
    shs = model.get_features
    if sh_modifier is not None:
        shs = shs + sh_modifier
    if depth:
        shs = None
        Rt = extrinsic.T[:3, :4]
        colors_precomp = (Rt[:3, :3] @ means3D.T + Rt[:, -1:]).T

    rendered_image, radii = rasterizer(means3D = means3D,
                                       means2D = means2D,
                                       shs = shs,
                                       colors_precomp = colors_precomp,
                                       opacities = opacity,
                                       scales = scales,
                                       rotations = rotations,
                                       cov3D_precomp = cov3D_precomp)
    visibility_filter = radii > 0
    torch.cuda.synchronize()
    if depth:
        rendered_image = rendered_image[-1]
    return rendered_image, screenspace_points, visibility_filter, radii


@torch.no_grad()
def get_model_params(model, preact: bool=False, device='cuda'):
    xyz = model.get_xyz.data.to(device)
    rotation = model.get_rotation.to(device)
    # get pre-activated params
    if preact:
        scale = model._scaling.data.to(device)
        opacity = model._opacity.data.to(device)
    else:
        scale = model.get_scaling.to(device)
        opacity = model.get_opacity.to(device)
    rgb_feat = model.get_features.to(device)
    return xyz, rotation, scale, opacity, rgb_feat


@torch.no_grad()
def sample_cameras(local_model,
                   global_metadatas: List[Dict[str, Any]],
                   max_cameras: int = 50,
                   far: int = 100) -> List[str]:
    height, width, fovx, fovy, viewmats = get_cameras_from_metadata(global_metadatas)
    xyz = local_model._xyz
    candidates = []
    viewpnts = []
    for i, (h, w, fx, fy, viewmat) in enumerate(zip(height, width, fovx, fovy, viewmats)):
        projmat = compute_projection(fx, fy, viewmat)
        mask = in_frustum_mask(xyz, viewmat, projmat, far)
        if mask.any():
            candidates.append(i)
            viewpnts.append(mask.sum().item())
    if len(candidates) > max_cameras:
        # sample {max_cameras} cameras based on #viewpnts
        sum_vpnts = sum(viewpnts)
        weights = [v / sum_vpnts for v in viewpnts]
        idx = np.random.choice(len(candidates), size=max_cameras, replace=False, p=weights)
        candidates = [candidates[i] for i in idx]
    return candidates



def get_cameras_from_metadata(metadatas: List[Dict[str, Any]],
                              indices: Optional[List[int]]=None,
                              colmap_fmt: bool=True) -> Tuple[List[float],
                                                              List[float],
                                                              List[float],
                                                              List[float],
                                                              torch.Tensor]:
    image_height = []
    image_width = []
    fovx = []
    fovy = []
    viewmats = []
    if indices is None:
        indices = range(len(metadatas))
    for i in indices:
        image_height.append(metadatas[i]['H'])
        image_width.append(metadatas[i]['W'])
        fx, fy, _, _ = metadatas[i]['intrinsics']
        if isinstance(fx, torch.Tensor):
            fovx.append(focal2fov(fx.item(), image_width[-1]))
            fovy.append(focal2fov(fy.item(), image_height[-1]))
        else:
            fovx.append(focal2fov(fx, image_width[-1]))
            fovy.append(focal2fov(fy, image_height[-1]))
        if colmap_fmt:
            viewmats.append(meganerf2colmap(metadatas[i]['c2w']))
        else:
            viewmats.append(metadatas[i]['c2w'])
    return image_height, image_width, fovx, fovy, torch.stack(viewmats).cuda()
