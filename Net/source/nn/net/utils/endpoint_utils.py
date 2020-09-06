from itertools import product

import torch
import torch.nn.functional as F

import Net.source.datasets.dataset_utils as du

from Net.source.utils.math_utils import sample_grid
from Net.source.utils.projection_utils import warp_keypoints_RBT, warp_keypoints_H

"""
Endpoint keys
"""
SCORE1 = 'score1'
SCORE2 = 'score2'

SAL_SCORE1 = 'sal_score1'
SAL_SCORE2 = 'sal_score2'

CONF_SCORE1 = 'conf_score1'
CONF_SCORE2 = 'conf_score2'

LOG_CONF_SCORE1 = 'log_conf_score1'
LOG_CONF_SCORE2 = 'log_conf_score2'

DESC1 = 'desc1'
DESC2 = 'desc2'

MODEL_INFO1 = 'model_info1'
MODEL_INFO2 = 'model_info2'

KP1 = 'kp1'
KP2 = 'kp2'

KP1_DESC = 'kp1_desc'
KP2_DESC = 'kp2_desc'

W_KP1 = 'w_kp1'
W_KP2 = 'w_kp2'
W_VIS_KP1_MASK = 'w_vis_kp1_mask'
W_VIS_KP2_MASK = 'w_vis_kp2_mask'

"""
Endpoint utils
"""


def select_kp(score, nms_kernel_size, top_k, scale_factor=1.0, return_values=False):
    """
    :param score: B x 1 x H x W :type torch.float
    :param nms_kernel_size :type float
    :param top_k :type int
    :param scale_factor: float
    :param return_values: bool
    :return B x N x 2 :type torch.long
    """
    b, _, h, w = score.size()

    # Apply nms
    score = nms(score, nms_kernel_size)

    # Extract maximum activation indices and convert them to keypoints
    score = score.view(b, -1)
    kp_values, flat_kp_ids = torch.topk(score, top_k)

    kp_ids = flat2grid(flat_kp_ids, w).float() * scale_factor + scale_factor / 2

    if return_values:
        return kp_ids, kp_values, flat_kp_ids

    else:
        return kp_ids


def sample_loc(loc, kp):
    """
    :param loc: B x 2 x H x W
    :param kp: B x N x 2
    """
    kp_grid = kp[:, :, [1, 0]].unsqueeze(1)

    kp_loc = sample_grid(loc, kp_grid).squeeze(2).permute(0, 2, 1)

    return kp_loc


def sample_descriptors(desc, kp, grid_size):
    """
    :param desc: B x C x H x W
    :param kp: B x N x 2
    :param grid_size: int
    :return B x N x C
    """
    kp_grid = kp[:, :, [1, 0]].unsqueeze(1) / grid_size

    kp_desc = F.normalize(sample_grid(desc, kp_grid).squeeze(2)).permute(0, 2, 1)

    return kp_desc


def warp_points(kp1, kp2, image1_shape, image2_shape, batch):

    if du.H12 in batch:
        H12, H21 = batch[du.H12].to(kp1.device), batch[du.H21].to(kp1.device)

        w_kp1, w_vis_kp1_mask = warp_keypoints_H(kp1, H12, image2_shape)
        w_kp2, w_vis_kp2_mask = warp_keypoints_H(kp2, H21, image1_shape)

    else:
        depth1, depth2, intrinsic1, intrinsic2, \
        extrinsic1, extrinsic2, shift_scale1, shift_scale2 = batch[du.DEPTH1].to(kp1.device), batch[du.DEPTH2].to(kp1.device), \
                                                             batch[du.INTRINSICS1].to(kp1.device), batch[du.INTRINSICS2].to(kp1.device), \
                                                             batch[du.EXTRINSICS1].to(kp1.device), batch[du.EXTRINSICS2].to(kp1.device), \
                                                             batch[du.SHIFT_SCALE1].to(kp1.device), batch[du.SHIFT_SCALE2].to(kp1.device)

        w_kp1, w_vis_kp1_mask = warp_keypoints_RBT(kp1, depth1, intrinsic1, extrinsic1, shift_scale1,
                                                   depth2, intrinsic2, extrinsic2, shift_scale2)
        w_kp2, w_vis_kp2_mask = warp_keypoints_RBT(kp2, depth2, intrinsic2, extrinsic2, shift_scale2,
                                                   depth1, intrinsic1, extrinsic1, shift_scale1)

    return w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask


"""
Support utils
"""


def nms(score, nms_kernel_size):
    """
    :param score: B x 1 x H x W
    :param nms_kernel_size: int
    :return B x 1 x H x W
    """
    padding = nms_kernel_size // 2
    max_score = F.max_pool2d(score, kernel_size=nms_kernel_size, stride=1, padding=padding)

    return score * (max_score == score).float()


def flat2grid(flat_ids, w):
    """
    :param flat_ids: B x C x N tensor of indices taken from flattened tensor of shape B x C x H x W
    :param w: Last dimension (W) of tensor from which indices were taken
    :return: B x C x N x 2 tensor of coordinates in input tensor B x C x H x W
    """
    y = flat_ids // w
    x = flat_ids - y * w

    y = y.unsqueeze(-1)
    x = x.unsqueeze(-1)

    return torch.cat((y, x), dim=-1)


def grid2flat(ids, w):
    """
    :param ids: B x N x 2, tensor of indices :type torch.long
    :param w: last dimension (W) of tensor of indices :type long
    """
    return w * ids[:, :, 0] + ids[:, :, 1]

# Legacy code

# def sample_neigh_desc(desc, kp, grid_size):
#     """
#     :param desc: B x C x H x W
#     :param kp: B x N x 2; y,x coordinates positioning
#     :return B x 8 x N x C
#     """
#     n, c = kp.shape[1], desc.shape[1]
#
#     # Generate 8 neighbouring points around each keypoint
#     neigh_kp = kp.unsqueeze(1)
#     shifts = list(set(product([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])) - {(0.0, 0.0)})
#     shifts = torch.tensor(shifts).unsqueeze(0).unsqueeze(-2).to(neigh_kp.device)
#     neigh_kp = neigh_kp + shifts
#
#     neigh_desc = sample_descriptors(desc, neigh_kp.view(-1, 8 * n, 2), grid_size).view(-1, 8, n, c)
#
#     return neigh_desc

# def localize_kp(score1, kp1):
#     """
#     :param score1: B x 1 X H x W :type torch.float
#     :param kp1: B x N x 2 :type torch.long
#     :return: B x N x 2 :type torch.float
#     """
#     b, c, _, w = score1.shape
#
#     dy_kernel = torch.tensor([[[[0., -1., 0.],
#                                 [0., 0., 0.],
#                                 [0., 1., 0.]]]]).to(score1.device)
#     dx_kernel = torch.tensor([[[[0., 0., 0.],
#                                 [-1., 0., 1.],
#                                 [0., 0., 0.]]]]).to(score1.device)
#
#     dyy_kernel = torch.tensor([[[[0., 1., 0.],
#                                  [0., -2., 0.],
#                                  [0., 1., 0.]]]]).to(score1.device)
#     dxy_kernel = torch.tensor([[[[-1., 0., -1.],
#                                  [0., 0., 0.],
#                                  [-1., 0., 1.]]]]).to(score1.device)
#     dxx_kernel = torch.tensor([[[[0., 0., 0.],
#                                  [1., -2., 1.],
#                                  [0., 0., 0.]]]]).to(score1.device)
#
#     dy = F.conv2d(score1, dy_kernel, padding=1) / 2
#     dx = F.conv2d(score1, dx_kernel, padding=1) / 2
#
#     dyy = F.conv2d(score1, dyy_kernel, padding=1)
#     dxy = F.conv2d(score1, dxy_kernel, padding=1) / 4
#     dxx = F.conv2d(score1, dxx_kernel, padding=1)
#
#     flat_kp1 = grid2flat(kp1, w)
#
#     kp_dy = torch.gather(dy.view(b, -1), -1, flat_kp1).unsqueeze(-1)
#     kp_dx = torch.gather(dx.view(b, -1), -1, flat_kp1).unsqueeze(-1)
#
#     kp_dyy = torch.gather(dyy.view(b, -1), -1, flat_kp1).unsqueeze(-1)
#     kp_dxy = torch.gather(dxy.view(b, -1), -1, flat_kp1).unsqueeze(-1)
#     kp_dxx = torch.gather(dxx.view(b, -1), -1, flat_kp1).unsqueeze(-1)
#
#     dD = torch.cat((kp_dy, kp_dx), dim=-1).view(-1, 2, 1)
#     H = torch.cat((kp_dyy, kp_dxy, kp_dxy, kp_dxx), dim=-1).view(-1, 2, 2)
#
#     singularity_mask = torch.det(H) > 1e-5
#
#     non_singular_x_hat, _ = torch.solve(-dD[singularity_mask], H[singularity_mask])
#
#     x_hat = torch.zeros(H.shape[0], 2, dtype=torch.float, device=H.device)
#     x_hat[singularity_mask] = non_singular_x_hat.view(-1, 2)
#
#     return kp1.float() + x_hat.view(b, -1, 2)

# h12, h21 = batch.get_homo(d.H12), batch.get_homo(d.H21)
#
# kp1_homo = batch.split_h(kp1)
# kp2_homo = batch.split_h(kp2)
#
# w_kp1_h = warp_points_h(kp1_homo, h12)
# w_kp2_h = warp_points_h(kp2_homo, h21)
#
# w_vis_kp1_mask_h = get_visibility_mask(batch.split_h(image2).shape, w_kp1_h)
# w_vis_kp2_mask_h = get_visibility_mask(batch.split_h(image1).shape, w_kp2_h)

# b, w = loc.shape[0], loc.shape[-1]

# flat_loc = loc.view(b, 2, -1).permute(0, 2, 1) # B X H * W x 2
# flat_kp = grid2flat(kp, w).unsqueeze(-1).repeat(1, 1, 2) # B x N x 2
#
# kp_loc = torch.gather(flat_loc, 1, flat_kp)

# def select_kp_by_thresh(score, nms_kernel_size, score_thresh):
#     # Apply nms
#     score = nms(score, nms_kernel_size)
#
#     # Extract maximum activations that are larger than a threshold
#     kp_mask = score > score_thresh
#     kp = kp_mask.nonzero()[:, 2:].unsqueeze(0)
#
#     return kp