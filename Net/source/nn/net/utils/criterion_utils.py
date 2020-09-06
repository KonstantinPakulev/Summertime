import torch
import torch.nn.functional as F

import Net.source.datasets.dataset_utils as du

from Net.source.core.wrapper import LOSS

from Net.source.utils.math_utils import create_coord_grid, calculate_distance_mat
from Net.source.nn.net.utils.endpoint_utils import select_kp

from Net.source.utils.projection_utils import warp_image_RBT, warp_coord_grid_RBT, warp_image_H, warp_coord_grid_H,\
    get_visibility_mask

"""
Loss keys
"""

DET_LOSS = 'det_loss'
DET_CONF_LOSS = 'det_conf_loss'

DESC_LOSS = 'des_loss'

EPIPOLAR_LOSS = 'epipolar_loss'
POSE_LOSS = 'pose_loss'

"""
Loss utility functions
"""


def warp_image(image1, image2, batch, scale_factor=1.0):
    if du.H12 in batch:
        H12, H21 = batch[du.H12].to(image1.device), batch[du.H21].to(image1.device)

        if scale_factor != 1.0:
            H12 = resize_homography(H12, scale_factor, True)
            H21 = resize_homography(H21, scale_factor, False)

        w_image1, w_vis_mask1 = warp_image_H(image2.shape, image1, H21)
        w_image2, w_vis_mask2 = warp_image_H(image1.shape, image2, H12)

    else:
        depth1, depth2, intrinsic1, intrinsic2, \
        extrinsic1, extrinsic2, shift_scale1, shift_scale2 = batch[du.DEPTH1].to(image1.device), batch[du.DEPTH2].to(image1.device), \
                                                             batch[du.INTRINSICS1].to(image1.device), batch[du.INTRINSICS2].to(image1.device), \
                                                             batch[du.EXTRINSICS1].to(image1.device), batch[du.EXTRINSICS2].to(image1.device), \
                                                             batch[du.SHIFT_SCALE1].to(image1.device), batch[du.SHIFT_SCALE2].to(image1.device)

        if scale_factor != 1.0:
            depth1 = F.interpolate(depth1, scale_factor=scale_factor, mode='bilinear')
            depth2 = F.interpolate(depth2, scale_factor=scale_factor, mode='bilinear')

            shift_scale1[:, 2:] *= scale_factor
            shift_scale2[:, 2:] *= scale_factor

        w_image1, w_vis_mask1 = warp_image_RBT(image1, depth2, intrinsic2, extrinsic2, shift_scale2, depth1,
                                               intrinsic1, extrinsic1, shift_scale1)
        w_image2, w_vis_mask2 = warp_image_RBT(image2, depth1, intrinsic1, extrinsic1, shift_scale1, depth2,
                                               intrinsic2, extrinsic2, shift_scale2)

    return w_image1, w_vis_mask1, w_image2, w_vis_mask2


def gaussian_filter(score, gauss_kernel_size, gauss_sigma: float):
    """
    :param score: N x 1 x H x W
    :param gauss_kernel_size: kernel size
    :param gauss_sigma: standard deviation
    """
    mu_x = mu_y = gauss_kernel_size // 2

    if gauss_sigma == 0:
        gauss_kernel = torch.zeros((1, 1, gauss_kernel_size, gauss_kernel_size), dtype=torch.float).to(score.device)
        gauss_kernel[0, 0, mu_y, mu_x] = 1.0
    else:
        x = torch.arange(gauss_kernel_size)[None, :].repeat(gauss_kernel_size, 1).float()
        y = torch.arange(gauss_kernel_size)[:, None].repeat(1, gauss_kernel_size).float()

        gauss_kernel = torch.exp(-((x - mu_x) ** 2 / (2 * gauss_sigma ** 2) + (y - mu_y) ** 2 / (2 * gauss_sigma ** 2)))
        gauss_kernel = gauss_kernel.view(1, 1, gauss_kernel_size, gauss_kernel_size).to(score.device)

    score = apply_kernel(score, gauss_kernel).clamp(min=0.0, max=1.0)

    return score


def prepare_gt_score(w_score2, nms_kernel_size, top_k, w_vis_mask2):
    """
    :param w_score2: B x 1 x H x W
    :param nms_kernel_size: int
    :param top_k: int
    :param w_vis_mask2: B x 1 x H x W
    :return B x 1 x H x W, B x N x 2
    """
    b, _, h, w = w_score2.size()

    flat_kp_ids, vis_kp_mask = select_visible_kp(w_score2, nms_kernel_size, top_k, w_vis_mask2)

    # Set maximum activations
    gt_score = torch.zeros_like(w_score2.view(b, -1)).to(w_score2.device)
    gt_score = gt_score.scatter(dim=-1, index=flat_kp_ids, src=vis_kp_mask.float()).view(b, 1, h, w)

    return gt_score


def create_w_desc_grid_RBT(image2_shape, grid_size, depth1, intrinsic1, extrinsic1, shift_scale1, depth2,
                           intrinsic2, extrinsic2, shift_scale2):
    """
    :param image2_shape: (b, 1, h, w)
    :param grid_size: size of the descriptor grid
    :param depth1: B x 1 x H x W :type torch.float
    :param intrinsic1: B x 3 x 3 :type torch.float
    :param extrinsic1: B x 4 x 4 :type torch.float
    :param shift_scale1: B x 4 :type tuple
    :param depth2: B x 1 x H x W :type torch.float
    :param intrinsic2: B x 3 x 3 :type torch.float
    :param extrinsic2: B x 4 x 4 :type torch.float
    :param shift_scale2: B x 4 :type tuple
    """
    b, _, h, w = image2_shape
    desc_shape = (b, 1, h // grid_size, w // grid_size)

    desc_grid1 = create_coord_grid(desc_shape, scale_factor=grid_size).to(depth1.device)
    w_desc_grid1, depth_mask1 = warp_coord_grid_RBT(desc_grid1, depth1, intrinsic1, extrinsic1, shift_scale1,
                                                    depth2, intrinsic2, extrinsic2, shift_scale2)

    w_desc_grid1 = w_desc_grid1.view(b, -1, 2)[:, :, [1, 0]]
    depth_mask1 = depth_mask1.view(b, -1)

    w_vis_desc_grid_mask1 = get_visibility_mask(image2_shape, w_desc_grid1) * depth_mask1

    return w_desc_grid1, w_vis_desc_grid_mask1


def create_w_desc_grid_H(image2_shape, grid_size, H12):
    """
    :param image2_shape: (b, 1, h, w)
    :param grid_size: size of the descriptor grid
    :param homo12: B x 3 x 3 :type torch.float
    """
    b, _, h, w = image2_shape
    desc_shape = (b, 1, h // grid_size, w // grid_size)

    desc_grid1 = create_coord_grid(desc_shape, scale_factor=grid_size).to(H12.device)
    w_desc_grid1 = warp_coord_grid_H(desc_grid1, H12)
    w_desc_grid1 = w_desc_grid1.view(b, -1, 2)[:, :, [1, 0]]

    w_vis_desc_grid_mask1 = get_visibility_mask(image2_shape, w_desc_grid1)

    return w_desc_grid1, w_vis_desc_grid_mask1


def create_w_desc_grid(image1_shape, image2_shape, batch, grid_size, device):
    """
    :param image1_shape: B x 1 x H x W :type torch.float
    :param image2_shape: B x 1 x H x W :type torch.float
    :param batch: :type CompositeBatch
    :param grid_size: :type int
    """

    if du.H12 in batch:
        H12, H21 = batch[du.H12].to(device), batch[du.H21].to(device)

        w_desc_grid1, w_vis_desc_grid_mask1 = create_w_desc_grid_H(image2_shape, grid_size, H12)
        w_desc_grid2, w_vis_desc_grid_mask2 = create_w_desc_grid_H(image1_shape, grid_size, H21)

    else:
        depth1, depth2, intrinsic1, intrinsic2, \
        extrinsic1, extrinsic2, shift_scale1, shift_scale2 = batch[du.DEPTH1].to(device), batch[du.DEPTH2].to(device), \
                                                             batch[du.INTRINSICS1].to(device), batch[du.INTRINSICS2].to(device), \
                                                             batch[du.EXTRINSICS1].to(device), batch[du.EXTRINSICS2].to(device), \
                                                             batch[du.SHIFT_SCALE1].to(device), batch[du.SHIFT_SCALE2].to(device)

        w_desc_grid1, w_vis_desc_grid_mask1 = create_w_desc_grid_RBT(image2_shape,
                                                                     grid_size, depth1, intrinsic1,
                                                                     extrinsic1, shift_scale1, depth2, intrinsic2,
                                                                     extrinsic2, shift_scale2)

        w_desc_grid2, w_vis_desc_grid_mask2 = create_w_desc_grid_RBT(image1_shape,
                                                                     grid_size, depth2, intrinsic2,
                                                                     extrinsic2, shift_scale2, depth1, intrinsic1,
                                                                     extrinsic1, shift_scale1)

    return w_desc_grid1, w_vis_desc_grid_mask1, w_desc_grid2, w_vis_desc_grid_mask2


def create_neigh_mask_ids(w_desc_grid1, desc_shape, grid_size):
    b, wc = desc_shape[0], desc_shape[-1]

    desc_grid2 = create_coord_grid(desc_shape, center=False, scale_factor=grid_size) \
        .view(b, -1, 2).to(w_desc_grid1.device)
    desc_grid2 = desc_grid2[..., [1, 0]]

    desc_grid_dist = calculate_distance_mat(w_desc_grid1, desc_grid2)
    neigh_mask_ids = desc_grid_dist.topk(6, dim=-1, largest=False)[1]

    return neigh_mask_ids


"""
Support utils
"""


def resize_homography(H, scale_factor, is12):
    if is12:
        scale_factor = 1.0 / scale_factor

    t = torch.tensor([[scale_factor, 0.0, 0.0],
                      [0.0, scale_factor, 0.0],
                      [0.0, 0.0, 1]]).unsqueeze(0)

    return H @ t


def apply_kernel(score, kernel):
    """
    :param score: N x 1 x H x W
    :param kernel: 1 x 1 x ks x ks
    :return:
    """
    score = F.conv2d(score, weight=kernel, padding=kernel.size(2) // 2)
    score = score.to(score.device)

    return score


def select_visible_kp(w_score2, nms_kernel_size, top_k, w_vis_mask2, return_flat=True, scale_factor=1.0):
    """
    :param w_score2: B x 1 x H x W
    :param nms_kernel_size: int
    :param top_k: int
    :param w_vis_mask2: B x 1 x H x W
    :param return_flat: bool
    :param scale_factor: float
    """
    b, _, h, w = w_score2.size()

    kp_ids, kp_values, flat_kp_ids = select_kp(w_score2, nms_kernel_size, top_k, scale_factor, True)

    # Determine the number of activations to leave based on the space occupied by the ground-truth data
    vis_top_k = (top_k * w_vis_mask2.view(b, -1).sum(dim=-1).float() / (h * w)).long().clamp(min=1)
    least_vis_kp_value = torch.gather(kp_values, 1, vis_top_k.view(-1, 1) - 1)
    vis_kp_mask = kp_values >= least_vis_kp_value  # B x N

    if return_flat:
        return flat_kp_ids, vis_kp_mask

    else:
        return kp_ids, vis_kp_mask


#  Legacy code

# def prepare_gt_score_old(score, nms_kernel_size, top_k):
#     """
#     :param score: B x 1 x H x W
#     :param nms_kernel_size: int
#     :param top_k: int
#     :return B x 1 x H x W, B x N x 2
#     """
#     b, _, h, w = score.size()
#
#     # Apply non-maximum suppression
#     score = nms(score, nms_kernel_size)
#
#     # Extract maximum activation indices
#     score = score.view(b, -1)
#     _, flat_ids = torch.topk(score, top_k)
#
#     # Select maximum activations
#     gt_score = torch.zeros_like(score).to(score.device)
#     gt_score = gt_score.scatter(dim=-1, index=flat_ids, value=1.0).view(b, 1, h, w)
#
#     return gt_score
