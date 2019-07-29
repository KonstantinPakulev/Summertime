import numpy as np
from skimage import transform

import torch
import torch.nn.functional as F


"""
Image manipulations
"""


def resize_image(image, output_size):
    """
    :param image: H x W x C
    :param output_size: (h_new, new_w)
    """
    h, w = image.shape[:2]

    if isinstance(output_size, int):
        if h > w:
            new_w = output_size
            new_h = output_size * h / w
        else:
            new_w = output_size * w / h
            new_h = output_size
    else:
        new_h, new_w = output_size

    new_w = int(new_w)
    new_h = int(new_h)
    image = transform.resize(image, (new_h, new_w), mode="constant")

    return image, (new_w / w, new_h / h)


def crop_image(image, rect):
    """
    :param image: H x W
    :param rect: (top, bottom, left, right)
    """
    top, bottom, left, right = rect

    return image[top: bottom, left: right]


"""
Homography functions
"""


def resize_homography(homography, ratio1=None, ratio2=None):
    """
    :param homography: 3 x 3
    :param ratio1: (new_w / w, new_h / h) of the first image
    :param ratio2: (new_w / w, new_h / h) of the second image
    """
    if ratio1 is not None:
        wr1, hr1 = ratio1
        t = np.mat([[1 / wr1, 0, 0],
                    [0, 1 / hr1, 0],
                    [0, 0, 1]], dtype=homography.dtype)

        homography = homography * t

    if ratio2 is not None:
        wr2, hr2 = ratio2
        t = np.mat([[wr2, 0, 0],
                    [0, hr2, 0],
                    [0, 0, 1]], dtype=homography.dtype)

        homography = t * homography

    return homography


def crop_homography(homography, rect1=None, rect2=None):
    """
    :param homography: 3 x 3
    :param rect1: (top, bottom, left, right) for the first image
    :param rect2: (top, bottom, left, right) for the second image
    """
    if rect1 is not None:
        top1, _, left1, _ = rect1

        t = np.mat([[1, 0, left1],
                    [0, 1, top1],
                    [0, 0, 1]], dtype=homography.dtype)

        homography = homography * t

    if rect2 is not None:
        top2, _, left2, _ = rect2

        t = np.mat([[1, 0, -left2],
                    [0, 1, -top2],
                    [0, 0, 1]], dtype=homography.dtype)

        homography = t * homography

    return homography


def create_coordinates_grid(output_size):
    """
    :param output_size: (n, c, h, w)
    """
    n, _, h, w = output_size

    gy, gx = torch.meshgrid([torch.arange(h), torch.arange(w)])
    gx = gx.float().unsqueeze(-1)
    gy = gy.float().unsqueeze(-1)

    grid = torch.cat((gx, gy), dim=-1)

    # Repeat grid for each batch
    grid = grid.unsqueeze(0)  # 1 x H x W x 2
    grid = grid.repeat(n, 1, 1, 1)  # B x H x W x 2

    return grid


def warp_coordinates_grid(grid, homography):
    """
    :param grid: N x H x W x 2
    :param homography: 3 x 3
    """
    n, h, w, _ = grid.size()

    # Convert grid to homogeneous coordinates
    ones = torch.ones((n, h, w, 1)).type_as(grid).to(grid.device)
    grid = torch.cat((grid, ones), dim=-1)  # N x H x W x 3

    # Flatten spatial dimensions
    grid = grid.view(n, -1, 3)  # B x H*W x 3
    grid = grid.permute(0, 2, 1)  # B x 3 x H*W

    # B x 3 x 3 matmul B x 3 x H*W => B x 3 x H*W
    w_grid = torch.matmul(homography, grid)
    w_grid = w_grid.permute(0, 2, 1)  # B x H*W x 3

    # Convert coordinates from homogeneous to cartesian
    w_grid = w_grid / (w_grid[:, :, 2].unsqueeze(-1) + 1e-8)  # B x H*W x 3
    w_grid = w_grid.view(n, h, w, -1)[:, :, :, :2]  # B x H x W x 2

    return w_grid


def warp_image(image, homography):
    """
    :param image: N x C x H x W
    :param homography: 3 x 3
    :return w_image: N x C x H x W
    """
    _, _, h, w = image.size()

    grid = create_coordinates_grid(image.size()).to(image.device)
    w_grid = warp_coordinates_grid(grid, homography)

    # Normalize coordinates in range [-1, 1]
    w_grid[:, :, :, 0] = w_grid[:, :, :, 0] / (w - 1) * 2 - 1
    w_grid[:, :, :, 1] = w_grid[:, :, :, 1] / (h - 1) * 2 - 1

    w_image = F.grid_sample(image, w_grid)  # N x C x H x W

    return w_image


"""
Kernel functions
"""


def apply_kernel(mask, kernel):
    """
    :param mask: N x 1 x H x W
    :param kernel: 1 x 1 x ks x ks
    :return:
    """
    _, _, ks, _ = kernel.size()

    kernel_mask = F.conv2d(mask, weight=kernel, padding=ks // 2)
    kernel_mask = kernel_mask.type_as(mask).to(mask.device)

    return kernel_mask


def erode_filter(mask):
    """
    :param mask: N x 1 x H x W
    """
    morph_ellipse_kernel = torch.tensor([[[[0, 0, 1, 0, 0],
                                           [1, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 1],
                                           [0, 0, 1, 0, 0]]]]).type_as(mask).to(mask.device)

    morphed_mask = apply_kernel(mask, morph_ellipse_kernel) / morph_ellipse_kernel.sum()
    morphed_mask = morphed_mask.ge(0.8).float()

    return morphed_mask


def dilate_filter(mask):
    """
    :param mask: N x 1 x H x W
    """
    dilate_kernel = torch.ones((1, 1, 3, 3)).type_as(mask).to(mask.device)

    dilated_mask = apply_kernel(mask, dilate_kernel).gt(0).float()

    return dilated_mask


def gaussian_filter(mask, ks, sigma: float):
    """
    :param mask: N x 1 x H x W
    :param ks: kernel size
    :param sigma: standard deviation
    """
    mu_x = mu_y = ks // 2

    if sigma == 0:
        gauss_kernel = torch.zeros((1, 1, ks, ks)).float()
        gauss_kernel[0, 0, mu_y, mu_x] = 1.0
    else:
        x = torch.arange(ks)[None, :].repeat(ks, 1).float()
        y = torch.arange(ks)[:, None].repeat(1, ks).float()
        gauss_kernel = torch.exp(-((x - mu_x) ** 2 / (2 * sigma ** 2) + (y - mu_y) ** 2 / (2 * sigma ** 2)))
        gauss_kernel = gauss_kernel.view(1, 1, ks, ks)

    gauss_kernel = gauss_kernel.to(mask.device)

    gauss_mask = apply_kernel(mask, gauss_kernel).clamp(min=0.0, max=1.0)

    return gauss_mask


"""
Score processing functions
"""


def nms(score, thresh: float, k_size):
    """
    :param score: B x 1 x H x W
    :param thresh: float
    :param k_size: int
    """
    _, _, h, w = score.size()

    score = torch.where(score < thresh, torch.zeros_like(score), score)

    pad_size = k_size // 2
    ps2 = pad_size * 2
    pad = [ps2, ps2, ps2, ps2, 0, 0]

    pad_det = F.pad(score, pad)

    slice_map = torch.tensor([], dtype=pad_det.dtype, device=pad_det.device)
    for i in range(k_size):
        for j in range(k_size):
            _slice = pad_det[:, :, i: h + ps2 + i, j: w + ps2 + j]
            slice_map = torch.cat((slice_map, _slice), 1)

    max_slice, _ = slice_map.max(dim=1, keepdim=True)
    center_map = slice_map[:, slice_map.size(1) // 2, :, :].unsqueeze(1)

    nms_mask = torch.ge(center_map, max_slice)
    nms_mask = nms_mask[:, :, pad_size: h + pad_size, pad_size: w + pad_size].type_as(score)

    score = score * nms_mask

    return score


def select_keypoints(score, thresh, k_size, top_k):
    """
    :param score: N x 1 x H x W
    :param thresh: float
    :param k_size: int
    :param top_k: int
    """
    n, c, h, w = score.size()
    flat = h * w

    # Apply nms
    score = nms(score, thresh, k_size)

    # Extract maximum activation indices
    score = score.view(n, c, flat)
    _, top_k_indices = torch.topk(score, top_k)

    # Select maximum activations
    top_score = torch.zeros_like(score).to(score.device)
    top_score[:, :, top_k_indices] = score[:, :, top_k_indices]

    top_score = top_score.view(n, c, h, w)
    keypoints = top_score.nonzero()

    return top_score, keypoints




