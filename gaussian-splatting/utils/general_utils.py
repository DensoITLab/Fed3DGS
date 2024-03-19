#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from PIL import Image

import torch
import sys
from datetime import datetime
import numpy as np
import random


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution, resample=Image.BOX)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def rotmat2qvec(R: torch.Tensor):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
    K = torch.Tensor([[Rxx - Ryy - Rzz, 0, 0, 0],
                      [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                      [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                      [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], torch.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def mul_quaternion(q1, q2):
    return torch.stack([- q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3] + q1[..., 0] * q2[..., 0],
                        q1[..., 0] * q2[..., 1] - q1[..., 3] * q2[..., 2] + q1[..., 2] * q2[..., 3] + q1[..., 1] * q2[..., 0],
                        q1[..., 3] * q2[..., 1] + q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0],
                        - q1[..., 2] * q2[..., 1] + q1[..., 1] * q2[..., 2] + q1[..., 0] * q2[..., 3] + q1[..., 3] * q2[..., 0]], -1)


def quaternion_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def rotate_vec(q, v):
    v = torch.cat([torch.zeros_like(v[..., :1]), v], -1)
    return mul_quaternion(mul_quaternion(q, v), quaternion_conjugate(q))[..., 1:]


def dist_quaternion(q1, q2):
    theta = 2 * torch.arccos((q1 * q2).sum(-1, keepdim=True).clamp(-1, 1))
    return theta


def quat_log_map(q):
    theta = torch.arccos(q[..., :1].clamp(-1, 1))
    l = q[..., 1:] / (q[..., 1:].norm(dim=-1, keepdim=True) + 1e-8)
    return torch.cat([torch.zeros_like(q[..., :1]), l * theta], -1)


def quat_exp_map(log_q):
    theta = log_q.norm(dim=-1, keepdim=True)
    l = log_q / (theta + 1e-8)
    w = torch.cos(theta)
    q = torch.cat([w, torch.sin(theta) * l[..., 1:]], -1)
    return torch.nn.functional.normalize(q, dim=-1)


def slerp(q1, q2, coeff):
    """shperical linear interpolation

    Args:
        q1, q2 (torch.Tensor): quaternion (qw, qx, qy, qz) that is a Tensor of shape (*, 4)

    Returns:
        q (torch.Tensor): quaternion (qw, qx, qy, qz) that is a Tensor of shape (*, 4)
    """
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)
    theta = 0.5 * dist_quaternion(q1, q2)
    inv_sin_theta = 1. / torch.sin(theta)
    q = inv_sin_theta * (torch.sin(coeff * theta) * q1 + torch.sin((1 - coeff) * theta) * q2)
    return q


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
