import os
import numpy as np
import pandas as pd
from skimage import io, color

import torch
from torch.utils.data import Dataset

from Net.utils.image_utils import resize_image, resize_homography, crop_image, crop_homography

# Available modes
TRAIN = 'train'
VALIDATE = 'validate'
VALIDATE_SHOW = 'validate_show'


class HPatchesDataset(Dataset):

    def __init__(self, root_path, csv_file, transform=None, include_originals=False):
        """
        :param root_path: Path to the dataset folder
        :param csv_file: The name of csv file with annotations
        :param transform: Transforms
        """

        self.root_path = root_path
        self.annotations = pd.read_csv(os.path.join(self.root_path, csv_file))
        self.transform = transform
        self.include_originals = include_originals

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, id):
        folder = self.annotations.iloc[id, 0]
        im1_name = self.annotations.iloc[id, 1]
        im2_name = self.annotations.iloc[id, 2]

        im1_path = os.path.join(self.root_path, folder, im1_name)
        im2_path = os.path.join(self.root_path, folder, im2_name)

        im1 = io.imread(im1_path)
        im2 = io.imread(im2_path)

        homo = np.asmatrix(self.annotations.iloc[id, 3:].values).astype(np.float).reshape(3, 3)

        item = {"im1": im1, "im2": im2, "homo": homo}

        if self.include_originals:
            item["orig1"] = im1.copy()
            item["orig2"] = im2.copy()

        if self.transform:
            item = self.transform(item)

        return item


class Grayscale(object):

    def __call__(self, item):
        item["im1"] = np.expand_dims(color.rgb2gray(item["im1"]), -1)
        item["im2"] = np.expand_dims(color.rgb2gray(item["im2"]), -1)
        return item


class Normalize(object):

    def __init__(self, mean, std):
        assert isinstance(mean, float)
        assert isinstance(std, float)

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample["im1"] = (sample["im1"] - self.mean) / self.std
        sample["im2"] = (sample["im2"] - self.mean) / self.std
        return sample


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, item):
        im1, im2, homo = (
            item["im1"],
            item["im2"],
            item["homo"]
        )

        im1, r1 = resize_image(im1, self.output_size)
        im2, r2 = resize_image(im2, self.output_size)

        homo = resize_homography(homo, r1, r2)

        item["im1"] = im1
        item["im2"] = im2
        item["homo"] = homo

        if "orig1" in item:
            item["orig1"], _ = resize_image(item["orig1"], self.output_size)
            item["orig2"], _ = resize_image(item["orig2"], self.output_size)

        return item


class RandomCrop:

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, item):
        im1, im2, homo = (
            item["im1"],
            item["im2"],
            item["homo"]
        )

        new_h, new_w = self.output_size

        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        if h1 <= new_h or w1 <= new_w:
            im1, r1 = resize_image(im1, self.output_size)
            homo = resize_homography(homo, r1=r1)
        else:
            top1 = np.random.randint(0, h1 - new_h)
            left1 = np.random.randint(0, w1 - new_w)
            bottom1 = top1 + int(new_h)
            right1 = left1 + int(new_w)

            rect1 = (top1, bottom1, left1, right1)

            im1 = crop_image(im1, rect1)
            homo = crop_homography(homo, rect1=rect1)

        if h2 <= new_h or w2 <= new_w:
            im2, r2 = resize_image(im2, self.output_size)
            homo = resize_homography(homo, r2=r2)
        else:
            top2 = np.random.randint(0, h2 - new_h)
            left2 = np.random.randint(0, w1 - new_w)
            bottom2 = top2 + int(new_h)
            right2 = left2 + int(new_w)

            rect2 = (top2, bottom2, left2, right2)

            im2 = crop_image(im2, rect2)
            homo = crop_homography(homo, rect2=rect2)

        item["im1"] = im1
        item["im2"] = im2
        item["homo"] = homo

        return item


class ToTensor:

    def __call__(self, item):
        im1, im2, homo = (
            item["im1"],
            item["im2"],
            item["homo"]
        )

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im1 = im1.transpose((2, 0, 1))
        im2 = im2.transpose((2, 0, 1))

        item["im1"] = torch.from_numpy(im1).float()
        item["im2"] = torch.from_numpy(im2).float()
        item["homo"] = torch.from_numpy(np.asarray(homo)).float()

        if "orig1" in item:
            orig1 = item["orig1"].transpose((2, 0, 1))
            orig2 = item["orig2"].transpose((2, 0, 1))

            item["orig1"] = torch.from_numpy(orig1)
            item["orig2"] = torch.from_numpy(orig2)

        return item
