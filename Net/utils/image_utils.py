import numpy as np
from skimage import transform

import torch
import torch.nn.functional as F


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


def resize_homography(homography, r1=None, r2=None):
    """
    :param homography: 3 x 3
    :param r1: (new_w / w, new_h / h) of the first image
    :param r2: (new_w / w, new_h / h) of the second image
    """

    if r1 is not None:
        wr1, hr1 = r1
        t = np.mat([[1 / wr1, 0, 0],
                 [0, 1 / hr1, 0],
                 [0, 0, 1]], dtype=homography.dtype)

        homography = homography * t

    if r2 is not None:
        wr2, hr2 = r2
        t = np.mat([[wr2, 0, 0],
                    [0, hr2, 0],
                    [0, 0, 1]], dtype=homography.dtype)

        homography = t * homography

    return homography


def crop_image(image, rect):
    """
    :param image: H x W
    :param rect: (top, bottom, left, right)
    """
    top, bottom, left, right = rect

    return image[top: bottom, left: right]


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
    ones = torch.ones((n, h, w, 1))
    grid = torch.cat((grid, ones), dim=-1) # N x H x W x 3

    # Flatten spatial dimensions
    grid = grid.view(n, -1, 3)  # B x H*W x 3
    grid = grid.permute(0, 2, 1)  # B x 3 x H*W

    grid = grid.type_as(homography).to(homography.device)

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

    grid = create_coordinates_grid(image.size())
    w_grid = warp_coordinates_grid(grid, homography)

    # Normalize coordinates in range [-1, 1]
    w_grid[:, :, :, 0] = w_grid[:, :, :, 0] / (w - 1) * 2 - 1
    w_grid[:, :, :, 1] = w_grid[:, :, :, 1] / (h - 1) * 2 - 1

    w_image = F.grid_sample(image.permute((0, )), w_grid)  # N x C x H x W

    return w_image




