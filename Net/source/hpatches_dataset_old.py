import os
import numpy as np
import pandas as pd
from skimage import io, color

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop

from Net.source.utils.image_utils import resize_image, resize_homography, crop_image, crop_homography

# Batch variables
IMAGE1 = 'image1'
IMAGE2 = 'image2'
HOMO12 = 'homo12'
HOMO21 = 'homo21'
S_IMAGE1 = 's_image1'
S_IMAGE2 = 's_image2'


class HPatchesDatasetOld(Dataset):

    def __init__(self, root_path, csv_file, transform=None, include_sources=False):
        """
        :param root_path: Path to the dataset folder
        :param csv_file: The name of csv file with annotations
        :param transform: Transforms
        """

        self.root_path = root_path
        self.annotations = pd.read_csv(os.path.join(self.root_path, csv_file))
        self.transform = transform
        self.include_sources = include_sources

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, id):
        folder = self.annotations.iloc[id, 0]
        im1_name = self.annotations.iloc[id, 1]
        im2_name = self.annotations.iloc[id, 2]

        im1_path = os.path.join(self.root_path, folder, im1_name)
        im2_path = os.path.join(self.root_path, folder, im2_name)

        image1 = io.imread(im1_path)
        image2 = io.imread(im2_path)

        homo12 = np.asmatrix(self.annotations.iloc[id, 3:].values).astype(np.float).reshape(3, 3)
        homo21 = homo12.I

        item = {IMAGE1: image1, IMAGE2: image2, HOMO12: homo12, HOMO21: homo21}

        if self.include_sources:
            item[S_IMAGE1] = image1.copy()
            item[S_IMAGE2] = image2.copy()

        if self.transform:
            item = self.transform(item)

        return item


class GrayscaleOld(object):

    def __call__(self, item):
        item[IMAGE1] = np.expand_dims(color.rgb2gray(item[IMAGE1]), -1)
        item[IMAGE2] = np.expand_dims(color.rgb2gray(item[IMAGE2]), -1)
        return item


class NormalizeOld(object):

    def __init__(self, mean, std):
        assert isinstance(mean, float)
        assert isinstance(std, float)

        self.mean = mean
        self.std = std

    def __call__(self, item):
        item[IMAGE1] = (item[IMAGE1] - self.mean) / self.std
        item[IMAGE2] = (item[IMAGE2] - self.mean) / self.std
        return item


class RescaleOld(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, item):
        image1, image2, homo12, homo21 = (
            item[IMAGE1],
            item[IMAGE2],
            item[HOMO12],
            item[HOMO21]
        )

        image1, ratio1 = resize_image(image1, self.output_size)
        image2, ratio2 = resize_image(image2, self.output_size)

        homo12 = resize_homography(homo12, ratio1, ratio2)
        homo21 = resize_homography(homo21, ratio2, ratio1)

        item[IMAGE1] = image1
        item[IMAGE2] = image2
        item[HOMO12] = homo12
        item[HOMO21] = homo21

        if S_IMAGE1 in item:
            item[S_IMAGE1], _ = resize_image(item[S_IMAGE1], self.output_size)
            item[S_IMAGE2], _ = resize_image(item[S_IMAGE2], self.output_size)

        return item


class RandomCropOld:

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, item):
        image1, image2, homo12, homo21 = (
            item[IMAGE1],
            item[IMAGE2],
            item[HOMO12],
            item[HOMO21]
        )

        new_h, new_w = self.output_size

        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        if h1 <= new_h or w1 <= new_w:
            image1, ratio1 = resize_image(image1, self.output_size)
            homo12 = resize_homography(homo12, ratio1=ratio1)
            homo21 = resize_homography(homo21, ratio2=ratio1)
        else:
            top1 = np.random.randint(0, h1 - new_h)
            left1 = np.random.randint(0, w1 - new_w)
            bottom1 = top1 + int(new_h)
            right1 = left1 + int(new_w)

            rect1 = (top1, bottom1, left1, right1)

            image1 = crop_image(image1, rect1)
            homo12 = crop_homography(homo12, rect1=rect1)
            homo21 = crop_homography(homo21, rect2=rect1)

        if h2 <= new_h or w2 <= new_w:
            image2, ratio2 = resize_image(image2, self.output_size)
            homo12 = resize_homography(homo12, ratio2=ratio2)
            homo21 = resize_homography(homo21, ratio1=ratio2)
        else:
            top2 = np.random.randint(0, h2 - new_h)
            left2 = np.random.randint(0, w1 - new_w)
            bottom2 = top2 + int(new_h)
            right2 = left2 + int(new_w)

            rect2 = (top2, bottom2, left2, right2)

            image2 = crop_image(image2, rect2)
            homo12 = crop_homography(homo12, rect2=rect2)
            homo21 = crop_homography(homo21, rect1=rect2)

        item[IMAGE1] = image1
        item[IMAGE2] = image2
        item[HOMO12] = homo12
        item[HOMO21] = homo21

        return item


class ToTensorOld:

    def __call__(self, item):
        image1, image2, homo12, homo21 = (
            item[IMAGE1],
            item[IMAGE2],
            item[HOMO12],
            item[HOMO21]
        )

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))

        item[IMAGE1] = torch.from_numpy(image1).float()
        item[IMAGE2] = torch.from_numpy(image2).float()
        item[HOMO12] = torch.from_numpy(np.asarray(homo12)).float()
        item[HOMO21] = torch.from_numpy(np.asarray(homo21)).float()

        if S_IMAGE1 in item:
            s_image1 = item[S_IMAGE1].transpose((2, 0, 1))
            s_image2 = item[S_IMAGE2].transpose((2, 0, 1))

            item[S_IMAGE1] = torch.from_numpy(s_image1)
            item[S_IMAGE2] = torch.from_numpy(s_image2)

        return item
