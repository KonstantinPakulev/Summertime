import os
import numpy as np
from numpy import random
import pandas as pd
from skimage import io, color

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from Net.source.utils.image_utils import resize_homography, crop_homography

# Batch variables
IMAGE1 = 'image1'
IMAGE2 = 'image2'
HOMO12 = 'homo12'
HOMO21 = 'homo21'
S_IMAGE1 = 's_image1'
S_IMAGE2 = 's_image2'


class HPatchesDataset(Dataset):

    def __init__(self, root_path, csv_file, item_transforms=None, include_sources=False):
        """
        :param root_path: Path to the dataset folder
        :param csv_file: The name of csv file with annotations
        :param item_transforms: Transforms for both homography and image
        :param include_sources:
        """

        self.root_path = root_path
        self.annotations = pd.read_csv(os.path.join(self.root_path, csv_file))
        self.item_transforms = item_transforms
        self.include_sources = include_sources

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, id):
        folder = self.annotations.iloc[id, 0]
        image1_name = self.annotations.iloc[id, 1]
        image2_name = self.annotations.iloc[id, 2]

        image1_path = os.path.join(self.root_path, folder, image1_name)
        image2_path = os.path.join(self.root_path, folder, image2_name)

        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)

        homo12 = np.asmatrix(self.annotations.iloc[id, 3:].values).astype(np.float).reshape(3, 3)
        homo21 = homo12.I

        item = {IMAGE1: image1, IMAGE2: image2, HOMO12: homo12, HOMO21: homo21}

        if self.include_sources:
            item[S_IMAGE1] = image1.copy()
            item[S_IMAGE2] = image2.copy()

        if self.item_transforms:
            item = self.item_transforms(item)

        return item


class PhotometricAugmentation(object):

    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast

        if brightness is not None:
            self.image_transforms = [transforms.ToPILImage(),
                                     transforms.ColorJitter(brightness=brightness, contrast=contrast),
                                     transforms.Grayscale()]
        else:
            self.image_transforms = [transforms.ToPILImage(),
                                     transforms.Grayscale()]

        self.image_transforms = transforms.Compose(self.image_transforms)

    def __call__(self, item):
        item[IMAGE1] = self.image_transforms(item[IMAGE1])
        item[IMAGE2] = self.image_transforms(item[IMAGE2])

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.to_pil_image(item[S_IMAGE1])
            item[S_IMAGE2] = F.to_pil_image(item[S_IMAGE2])

        return item


class ResizeItem(object):

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


class CropItem(object):

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


class FinishAugmentation(object):

    def __init__(self, mean, std):
        if mean is not None:
            self.image_transforms = [transforms.ToTensor(),
                                     transforms.Normalize([mean], [std])]
        else:
            self.image_transforms = [transforms.ToTensor()]
        self.image_transforms = transforms.Compose(self.image_transforms)

    def __call__(self, item):
        item[IMAGE1] = self.image_transforms(item[IMAGE1])
        item[IMAGE2] = self.image_transforms(item[IMAGE2])

        item[HOMO12] = torch.from_numpy(np.asarray(item[HOMO12])).float()
        item[HOMO21] = torch.from_numpy(np.asarray(item[HOMO21])).float()

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.to_tensor(item[S_IMAGE1])
            item[S_IMAGE2] = F.to_tensor(item[S_IMAGE2])

        return item
