import numpy as np

import torch

import torchvision.transforms.functional as F

import Net.source.datasets.dataset_utils as du


class MegaDepthToPILImage:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_pil_image(item[du.IMAGE1])
        item[du.IMAGE2] = F.to_pil_image(item[du.IMAGE2])

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.to_pil_image(item[du.S_IMAGE1])
            item[du.S_IMAGE2] = F.to_pil_image(item[du.S_IMAGE2])

        item[du.DEPTH1] = F.to_pil_image(item[du.DEPTH1], mode='F')
        item[du.DEPTH2] = F.to_pil_image(item[du.DEPTH2], mode='F')

        return item


class MegaDepthToTensor:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_tensor(item[du.IMAGE1])
        item[du.IMAGE2] = F.to_tensor(item[du.IMAGE2])

        item[du.DEPTH1] = F.to_tensor(np.array(item[du.DEPTH1]))
        item[du.DEPTH2] = F.to_tensor(np.array(item[du.DEPTH2]))

        item[du.EXTRINSICS1] = torch.from_numpy(item[du.EXTRINSICS1].astype(np.float32, copy=False))
        item[du.EXTRINSICS2] = torch.from_numpy(item[du.EXTRINSICS2].astype(np.float32, copy=False))

        item[du.INTRINSICS1] = torch.from_numpy(item[du.INTRINSICS1].astype(np.float32, copy=False))
        item[du.INTRINSICS2] = torch.from_numpy(item[du.INTRINSICS2].astype(np.float32, copy=False))

        item[du.SHIFT_SCALE1] = torch.from_numpy(item[du.SHIFT_SCALE1].astype(np.float32, copy=False))
        item[du.SHIFT_SCALE2] = torch.from_numpy(item[du.SHIFT_SCALE2].astype(np.float32, copy=False))

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.to_tensor(item[du.S_IMAGE1])
            item[du.S_IMAGE2] = F.to_tensor(item[du.S_IMAGE2])

        return item


class MegaDepthToGrayScale:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_grayscale(item[du.IMAGE1])
        item[du.IMAGE2] = F.to_grayscale(item[du.IMAGE2])

        return item


class MegaDepthCrop:

    def __init__(self, crop):
        self.crop = crop

    def __call__(self, item):
        if isinstance(self.crop, MegaDepthSharedAreaCrop):
            rect1 = self.crop.get_rect(item[du.DEPTH1])
            rect2 = self.crop.get_rect(item[du.DEPTH2])

        else:
            rect1 = self.crop.get_rect(item[du.IMAGE1])
            rect2 = self.crop.get_rect(item[du.IMAGE2])

        item[du.IMAGE1] = F.crop(item[du.IMAGE1], *rect1)
        item[du.IMAGE2] = F.crop(item[du.IMAGE2], *rect2)

        item[du.DEPTH1] = F.crop(item[du.DEPTH1], *rect1)
        item[du.DEPTH2] = F.crop(item[du.DEPTH2], *rect2)

        item[du.SHIFT_SCALE1] = crop_shift_scale(item[du.SHIFT_SCALE1], rect1)
        item[du.SHIFT_SCALE2] = crop_shift_scale(item[du.SHIFT_SCALE2], rect2)

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.crop(item[du.S_IMAGE1], *rect1)
            item[du.S_IMAGE2] = F.crop(item[du.S_IMAGE2], *rect2)

        return item


class MegaDepthResize:

    def __init__(self, resize):
        self.resize = resize

    def __call__(self, item):
        size1, scale1 = self.resize.get_size_scale(item[du.IMAGE1])
        size2, scale2 = self.resize.get_size_scale(item[du.IMAGE2])

        item[du.IMAGE1] = F.resize(item[du.IMAGE1], size1)
        item[du.IMAGE2] = F.resize(item[du.IMAGE2], size2)

        item[du.DEPTH1] = F.resize(item[du.DEPTH1], size1)
        item[du.DEPTH2] = F.resize(item[du.DEPTH2], size2)

        item[du.SHIFT_SCALE1] = resize_shift_scale(item[du.SHIFT_SCALE1], scale1)
        item[du.SHIFT_SCALE2] = resize_shift_scale(item[du.SHIFT_SCALE2], scale2)

        if du.S_IMAGE1 in item:
            item[du.S_IMAGE1] = F.resize(item[du.S_IMAGE1], size1)
            item[du.S_IMAGE2] = F.resize(item[du.S_IMAGE2], size2)

        return item


class MegaDepthSharedAreaCrop(du.CropBase):

    def get_rect(self, depth):
        depth = np.array(depth)

        column_mask = depth.sum(axis=-1) > 0
        row_mask = depth.sum(axis=-2) > 0

        h_loc = column_mask.nonzero()[0]
        w_loc = row_mask.nonzero()[0]

        offset_h = h_loc[0]
        new_height = h_loc[-1] - offset_h

        offset_w = w_loc[0]
        new_width = w_loc[-1] - offset_w

        rect = (offset_h, offset_w, new_height, new_width)

        return rect


"""
Support utils
"""


def crop_shift_scale(shift_scale, rect):
    shift_scale[:2] += np.array(rect[:2]) / shift_scale[2:]
    return shift_scale


def resize_shift_scale(shift_scale, scale_factor):
    shift_scale[2:] *= scale_factor
    return shift_scale
