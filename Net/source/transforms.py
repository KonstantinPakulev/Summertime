import torch

from numpy import random

import torchvision.transforms.functional as F

from Net.source.hpatches_dataset import *
from Net.source.utils.image_utils import resize_homography, crop_homography


class ToPILImage(object):

    def __call__(self, item):
        item[IMAGE1] = F.to_pil_image(item[IMAGE1])
        item[IMAGE2] = F.to_pil_image(item[IMAGE2])

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.to_pil_image(item[S_IMAGE1])
            item[S_IMAGE2] = F.to_pil_image(item[S_IMAGE2])

        return item


class ColorJitter(object):

    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, item):
        item[IMAGE1] = F.adjust_brightness(item[IMAGE1], self.brightness)
        item[IMAGE2] = F.adjust_contrast(item[IMAGE2], self.contrast)

        return item


class GrayScale(object):

    def __call__(self, item):
        item[IMAGE1] = F.to_grayscale(item[IMAGE1])
        item[IMAGE2] = F.to_grayscale(item[IMAGE2])

        return item


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, item):
        ratio1 = (self.size[1] / item[IMAGE1].size[0], self.size[0] / item[IMAGE1].size[1])
        ratio2 = (self.size[1] / item[IMAGE2].size[0], self.size[0] / item[IMAGE2].size[1])

        item[HOMO12] = resize_homography(item[HOMO12], ratio1, ratio2)
        item[HOMO21] = resize_homography(item[HOMO21], ratio2, ratio1)

        item[IMAGE1] = F.resize(item[IMAGE1], self.size)
        item[IMAGE2] = F.resize(item[IMAGE2], self.size)

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.resize(item[S_IMAGE1], self.size)
            item[S_IMAGE2] = F.resize(item[S_IMAGE2], self.size)

        return item


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def get_params(self, image):
        w, h = image.size

        i = random.randint(0, h - self.size[0])
        j = random.randint(0, w - self.size[1])

        rect = (i, i + self.size[0], j, j + self.size[1])

        return i, j, rect

    def __call__(self, item):
        i1, j1, rect1 = self.get_params(item[IMAGE1])
        i2, j2, rect2 = self.get_params(item[IMAGE2])

        item[HOMO12] = crop_homography(item[HOMO12], rect1=rect1)
        item[HOMO12] = crop_homography(item[HOMO12], rect2=rect2)

        item[HOMO21] = crop_homography(item[HOMO21], rect2=rect1)
        item[HOMO21] = crop_homography(item[HOMO21], rect1=rect2)

        item[IMAGE1] = F.crop(item[IMAGE1], i1, j1, self.size[0], self.size[1])
        item[IMAGE2] = F.crop(item[IMAGE2], i2, j2, self.size[0], self.size[1])

        return item


class ToTensor(object):

    def __call__(self, item):
        item[IMAGE1] = F.to_tensor(item[IMAGE1])
        item[IMAGE2] = F.to_tensor(item[IMAGE2])

        item[HOMO12] = torch.from_numpy(np.asarray(item[HOMO12])).float()
        item[HOMO21] = torch.from_numpy(np.asarray(item[HOMO21])).float()

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.to_tensor(item[S_IMAGE1])
            item[S_IMAGE2] = F.to_tensor(item[S_IMAGE2])

        return item
