import numpy as np

import torch

import torchvision.transforms.functional as F

import Net.source.datasets.dataset_utils as du

from Net.source.datasets.megadepth.megadepth_transforms import crop_shift_scale, resize_shift_scale


class AachenToPILImage:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_pil_image(item[du.IMAGE1])

        return item


class AachenToTensor:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_tensor(item[du.IMAGE1])

        item[du.SHIFT_SCALE1] = torch.from_numpy(item[du.SHIFT_SCALE1].astype(np.float32, copy=False))

        return item


class AachenToGrayScale:

    def __call__(self, item):
        item[du.IMAGE1] = F.to_grayscale(item[du.IMAGE1])

        return item


class AachenCrop:

    def __init__(self, crop):
        self.crop = crop

    def __call__(self, item):
        rect1 = self.crop.get_crop_rect(item[du.IMAGE1])

        item[du.IMAGE1] = F.crop(item[du.IMAGE1], *rect1)

        item[du.SHIFT_SCALE1] = crop_shift_scale(item[du.SHIFT_SCALE1], rect1)

        return item


class AachenResize:

    def __init__(self, resize):
        self.resize = resize

    def __call__(self, item):
        size1, scale1 = self.resize.get_size_scale(item[du.IMAGE1])

        item[du.IMAGE1] = F.resize(item[du.IMAGE1], size1)

        item[du.SHIFT_SCALE1] = resize_shift_scale(item[du.SHIFT_SCALE1], scale1)

        return item

