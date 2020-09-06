import numpy as np

import torch

import torchvision.transforms.functional as F

import Net.source.datasets.dataset_utils as du


class HPatchesToPILImage:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_pil_image(item[du.IMAGE1])
        item[du.IMAGE2] = F.to_pil_image(item[du.IMAGE2])

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.to_pil_image(item[du.S_IMAGE1])
            item[du.S_IMAGE2] = F.to_pil_image(item[du.S_IMAGE2])

        return item


class HPatchesToTensor:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_tensor(item[du.IMAGE1])
        item[du.IMAGE2] = F.to_tensor(item[du.IMAGE2])

        item[du.H12] = torch.from_numpy(item[du.H12]).float()
        item[du.H21] = torch.from_numpy(item[du.H21]).float()

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.to_tensor(item[du.S_IMAGE1])
            item[du.S_IMAGE2] = F.to_tensor(item[du.S_IMAGE2])

        return item


class HPatchesGrayScale:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_grayscale(item[du.IMAGE1])
        item[du.IMAGE2] = F.to_grayscale(item[du.IMAGE2])

        return item


class HPatchesCrop:

    def __init__(self, crop):
        self.crop = crop

    def __call__(self, item):
        rect1 = self.crop.get_rect(item[du.IMAGE1])
        rect2 = self.crop.get_rect(item[du.IMAGE2])

        item[du.IMAGE1] = F.crop(item[du.IMAGE1], *rect1)
        item[du.IMAGE2] = F.crop(item[du.IMAGE2], *rect2)

        item[du.H12] = crop_homography(item[du.H12], rect1, rect2)
        item[du.H21] = crop_homography(item[du.H21], rect2, rect1)

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.crop(item[du.S_IMAGE1], *rect1)
            item[du.S_IMAGE2] = F.crop(item[du.S_IMAGE2], *rect2)

        return item


class HPatchesResize:

    def __init__(self, resize):
        self.resize = resize

    def __call__(self, item):
        size1, scale1 = self.resize.get_size_scale(item[du.IMAGE1])
        size2, scale2 = self.resize.get_size_scale(item[du.IMAGE2])

        item[du.IMAGE1] = F.resize(item[du.IMAGE1], size1)
        item[du.IMAGE2] = F.resize(item[du.IMAGE2], size2)

        item[du.H12] = resize_homography(item[du.H12], scale1, scale2)
        item[du.H21] = resize_homography(item[du.H21], scale2, scale1)

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.resize(item[du.S_IMAGE1], size1)
            item[du.S_IMAGE2] = F.resize(item[du.S_IMAGE2], size2)

        return item


"""
Support utils
"""


def crop_homography(h, rect1=None, rect2=None):
    """
    :param h: 3 x 3
    :param rect1: (top, left, bottom, right) for the first image
    :param rect2: (top, left, bottom, right) for the second image
    """
    if rect1 is not None:
        top1, left1 = rect1[:2]

        t = np.mat([[1, 0, left1],
                    [0, 1, top1],
                    [0, 0, 1]], dtype=h.dtype)

        h = h * t

    if rect2 is not None:
        top2, left2 = rect2[:2]

        t = np.mat([[1, 0, -left2],
                    [0, 1, -top2],
                    [0, 0, 1]], dtype=h.dtype)

        h = t * h

    return h


def resize_homography(h, scale_factor1=None, scale_factor2=None):
    """
    :param h: 3 x 3
    :param scale_factor1: new_size / size of the first image :type numpy array
    :param scale_factor2: new_size / size of the second image :type numpy array
    """
    if scale_factor1 is not None:
        if np.ndim(scale_factor1) == 0:
            wr1 = scale_factor1
            hr1 = scale_factor1

        else:
            wr1, hr1 = scale_factor1

        t = np.mat([[1 / wr1, 0, 0],
                    [0, 1 / hr1, 0],
                    [0, 0, 1]], dtype=h.dtype)

        h = h * t

    if scale_factor2 is not None:
        if np.ndim(scale_factor2) == 0:
            wr2 = scale_factor2
            hr2 = scale_factor2

        else:
            wr2, hr2 = scale_factor2

        t = np.mat([[wr2, 0, 0],
                    [0, hr2, 0],
                    [0, 0, 1]], dtype=h.dtype)

        h = t * h

    return h
