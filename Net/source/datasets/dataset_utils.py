import numpy as np

import numpy.random as random

from abc import ABC, abstractmethod

import torch

from torch.utils.data.sampler import Sampler

"""
Datasets
"""
# TODO. That key may not be necessary
DATASET_NAME = 'dataset_name'

HPATCHES_VIEW = 'hpatches_view'
HPATCHES_ILLUM = 'hpatches_illum'

MEGADEPTH = 'megadepth'

AACHEN = 'aachen'

"""
Config keys
"""
# Dataset keys
DATASET_ROOT = 'dataset_root'
SCENE_INFO_ROOT = 'scene_info_root'
CSV_PATH = 'csv_path'
CSV_WARP_PATH = 'csv_warp_path'
TO_GRAYSCALE = 'to_grayscale'
RESIZE = 'resize'
HEIGHT = 'height'
WIDTH = 'width'
SOURCES = 'sources'

# Loader keys
BATCH_SIZE = 'batch_size'
NUM_SAMPLES = 'num_samples'
SHUFFLE = 'shuffle'
NUM_WORKERS = 'num_workers'

"""
Batch keys
"""
SCENE_NAME = 'scene_name'
IMAGE1_NAME = 'image1_name'
IMAGE2_NAME = 'image2_name'
IMAGE1 = 'image1'
IMAGE2 = 'image2'
DEPTH1 = 'depth1'
DEPTH2 = 'depth2'

ID1 = 'id1'
ID2 = 'id2'

EXTRINSICS1 = 'extrinsics1'
INTRINSICS1 = 'intrinsics1'
EXTRINSICS2 = 'extrinsics2'
INTRINSICS2 = 'intrinsics2'

SHIFT_SCALE1 = 'shift_scale1'
SHIFT_SCALE2 = 'shift_scale2'

H12 = 'h12'
H21 = 'h21'

S_IMAGE1 = 's_image1'
S_IMAGE2 = 's_image2'


"""
Data retrieval and processing utils 
"""


class CropBase(ABC):

    @abstractmethod
    def get_rect(self, image):
        ...


class ParityCrop(CropBase):

    def __init__(self, parity_factor):
        self.parity_factor = parity_factor

    def get_rect(self, image):
        if image.size[1] % self.parity_factor != 0:
            new_height = (image.size[1] // self.parity_factor) * self.parity_factor
            offset_h = int(round((image.size[1] - new_height) / 2.))
        else:
            offset_h = 0
            new_height = image.size[1]

        if image.size[0] % self.parity_factor != 0:
            new_width = (image.size[0] // self.parity_factor) * self.parity_factor
            offset_w = int(round((image.size[0] - new_width) / 2.))
        else:
            offset_w = 0
            new_width = image.size[0]

        rect = (offset_h, offset_w, new_height, new_width)

        return rect


class CentralCrop(CropBase):

    def __init__(self, size, is_train):
        """
        :param size: (h, w)
        """
        self.size = size
        self.is_train = is_train

    def get_rect(self, image):
        if image.size[0] > image.size[1] or self.is_train:
            new_height = self.size[0]
            new_width = self.size[1]
        else:
            new_height = self.size[1]
            new_width = self.size[0]

        offset_h = int(round((image.size[1] - new_height) / 2.))
        offset_w = int(round((image.size[0] - new_width) / 2.))

        rect = (offset_h, offset_w, new_height, new_width)

        return rect


class RandomCrop(CropBase):

    def __init__(self, size):
        """
        :param size: (h, w)
        """
        self.size = size

    def get_rect(self, image):
        offset_h = random.randint(0, image.size[1] - self.size[0] + 1)
        offset_w = random.randint(0, image.size[0] - self.size[1] + 1)

        rect = (offset_h, offset_w, self.size[0], self.size[1])

        return rect


class ResizeBase(ABC):

    @abstractmethod
    def get_size_scale(self, image):
        """
        :param image: PILImage. Note that it's size argument returns (w, h)
        """
        ...


class Resize(ResizeBase):

    def __init__(self, size):
        """
        :param size: (h, w)
        """
        self.size = size

    def get_size_scale(self, image):
        scale = np.array((self.size[0] / image.size[1], self.size[1] / image.size[0]))

        return self.size, scale


class FactorResize(ResizeBase):

    def __init__(self, resize_factor):
        self.resize_factor = resize_factor

    def get_size_scale(self, image):
        new_size = np.array(image.size[::-1]) * self.resize_factor

        return new_size, self.resize_factor


class AspectResize(ResizeBase):

    def __init__(self, size, is_train):
        """
        :param size: (h, w)
        """
        self.size = size
        self.is_train = is_train

    def get_size_scale(self, image):
        if self.is_train:
            # During training (and validation) images in a batch can only have horizontal orientation, so
            # if orientations do not align we transform it in a way to capture more vertical area of the input image
            if image.size[0] > image.size[1]:
                new_ar = float(self.size[1]) / float(self.size[0])
                ar = float(image.size[0]) / float(image.size[1])

                if ar >= new_ar:
                    new_size = min(self.size)
                    scale_factor = np.array(new_size / image.size[1])

                else:
                    scale_factor = np.array(self.size[1] / image.size[0])
                    new_size = (int(image.size[1] * scale_factor), self.size[1])

            else:
                new_size = max(self.size)
                scale_factor = np.array(new_size / min(image.size))

        else:
            # During test there is only one image in a batch, so the restriction above is removed
            new_size = self.size[0]
            scale_factor = np.array(self.size[0] / min(image.size))

        return new_size, scale_factor


class DatasetSubsetSampler(Sampler):

    def __init__(self, data_source, num_samples, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = num_samples
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        permutation = (torch.randperm(n).tolist() if self.shuffle else torch.arange(n).tolist())[:self.num_samples]
        return iter(permutation)

    def __len__(self):
        return self.num_samples


# Legacy code

# class CompositeBatch:
#
#     def __init__(self, h, r3, device):
#         self._h = h
#         self._r3 = r3
#
#         self.device = device
#
#         self.joint_index = len(self._h[IMAGE1]) if IMAGE1 in self._h else 0
#
#         self._is_homo = IMAGE1 in self._h
#         self._is_r3 = IMAGE1 in self._r3
#
#     @property
#     def is_h(self):
#         return self._is_homo
#
#     @property
#     def is_r3(self):
#         return self._is_r3
#
#     def get_homo(self, key):
#         return self._h[key].to(self.device)
#
#     def get_r3(self, key):
#         return self._r3[key].to(self.device)
#
#     def get(self, key):
#         joint_tensor = self.join(self._h.get(key), self._r3.get(key))
#         return joint_tensor.to(self.device) \
#             if joint_tensor is not None and not isinstance(joint_tensor, list) else joint_tensor
#
#     def split_h(self, tensor):
#         return tensor[:self.joint_index]
#
#     def split_r3(self, tensor):
#         return tensor[self.joint_index:]
#
#     def join(self, tensor1, tensor2):
#         joint_tensor = None
#
#         if self.is_h:
#             joint_tensor = tensor1
#
#         if self.is_r3:
#             if joint_tensor is not None:
#                 joint_tensor = torch.cat([joint_tensor, tensor2])
#             else:
#                 joint_tensor = tensor2
#
#         return joint_tensor


# class TwoDatasetsCollate:
#
#     def __init__(self, device):
#         self.device = device
#
#     def __call__(self, batch):
#         batch_homo = []
#         batch_r3 = []
#
#         for elem in batch:
#             (batch_homo if H12 in elem.keys() else batch_r3).append(elem)
#
#         t_batch_homo = default_collate(batch_homo) if len(batch_homo) != 0 else {}
#         t_batch_r3 = default_collate(batch_r3) if len(batch_r3) != 0 else {}
#
#         return CompositeBatch(t_batch_homo, t_batch_r3, self.device)
# class ColorJitter(object):
#
#     def __init__(self, brightness=0.1, contrast=0.1):
#         self.brightness = brightness
#         self.contrast = contrast
#
#     def __call__(self, item):
#         brightness_factor = random.uniform(max(1 - self.brightness, 0), 1 + self.brightness)
#         contrast_factor = random.uniform(max(1 - self.contrast, 0), 1 + self.contrast)
#
#         transforms = [Lambda(lambda image: F.adjust_brightness(image, brightness_factor)),
#                       Lambda(lambda image: F.adjust_contrast(image, contrast_factor))]
#         random.shuffle(transforms)
#         transforms = Compose(transforms)
#
#         item[d.IMAGE1] = transforms(item[d.IMAGE1])
#         item[d.IMAGE2] = transforms(item[d.IMAGE2])
#
#         return item
#
#
# class Normalize(object):
#
#     def __init__(self, mean, std):
#         self.mean = np.array(mean)
#         self.std = np.array(std)
#
#     def __call__(self, item):
#         item[d.IMAGE1] = item[d.IMAGE1] / 255.0
#         item[d.IMAGE2] = item[d.IMAGE2] / 255.0
#
#         item[d.IMAGE1] = (item[d.IMAGE1] - self.mean.reshape([1, 1, 3])) / self.std.reshape([1, 1, 3])
#         item[d.IMAGE2] = (item[d.IMAGE2] - self.mean.reshape([1, 1, 3])) / self.std.reshape([1, 1, 3])
#
#         return item
# from torchvision.transforms.transforms import Lambda, Compose


# class RandomWarp(object):
#
#     def __init__(self, perspective=True, scaling=True, rotation=True, translation=True,
#                  n_scales=5, n_angles=5, scaling_amplitude=0.2,
#                  perspective_amplitude_x=0.2, perspective_amplitude_y=0.2,
#                  patch_ratio=0.85, max_angle=math.pi / 16):
#         self.perspective = perspective
#         self.scaling = scaling
#         self.rotation = rotation
#         self.translation = translation
#         self.n_scales = n_scales
#         self.n_angles = n_angles
#         self.scaling_amplitude = scaling_amplitude
#         self.perspective_amplitude_x = perspective_amplitude_x
#         self.perspective_amplitude_y = perspective_amplitude_y
#         self.patch_ratio = patch_ratio
#         self.max_angle = max_angle
#
#     def __call__(self, item):
#         item[d.H12] = np.asmatrix(sample_homography(item[d.IMAGE1].shape[1::-1],
#                                                     self.perspective, self.scaling, self.rotation, self.translation,
#                                                     self.n_scales, self.n_angles, self.scaling_amplitude,
#                                                     self.perspective_amplitude_x, self.perspective_amplitude_y,
#                                                     self.patch_ratio, self.max_angle))
#         item[d.H21] = item[d.H12].I
#
#         item[d.IMAGE2] = cv2.warpPerspective(item[d.IMAGE1], item[d.H12], item[d.IMAGE1].shape[1::-1])
#
#         if d.S_IMAGE1 in item:
#             item[d.S_IMAGE2] = item[d.IMAGE2].copy()
#
#         return item
#
#
# class RandomCrop(object):
#

#
#     def __call__(self, item):
#         i1, j1, rect1 = self.get_params(item[d.IMAGE1])
#         item[d.IMAGE1] = F.crop(item[d.IMAGE1], i1, j1, self.size[0], self.size[1])
#
#         if d.IMAGE2 in item:
#             i2, j2, rect2 = self.get_params(item[d.IMAGE2])
#             item[d.IMAGE2] = F.crop(item[d.IMAGE2], i2, j2, self.size[0], self.size[1])
#
#         if d.DEPTH1 in item:
#             item[d.DEPTH1] = F.crop(item[d.DEPTH1], i1, j1, self.size[0], self.size[1])
#             item[d.DEPTH2] = F.crop(item[d.DEPTH2], i2, j2, self.size[0], self.size[1])
#
#         if d.SHIFT_SCALE1 in item:
#             # y, x shift
#             item[d.SHIFT_SCALE1][:2] += np.array([i1, j1]) / item[d.SHIFT_SCALE1][2:]
#
#         if d.SHIFT_SCALE2 in item:
#             # y, x shift
#             item[d.SHIFT_SCALE2][:2] += np.array([i2, j2]) / item[d.SHIFT_SCALE2][2:]
#
#         if d.H12 in item:
#             item[d.H12] = crop_homography(item[d.H12], rect1, rect2)
#             item[d.H21] = crop_homography(item[d.H21], rect2, rect1)
#
#         return item

#     item_transforms = get_test_transformation(d_key, d_config)
#     dataset = AachenDataset.from_config(d_config, item_transforms)
# if d_key == d.MEGADEPTH:
#     item_transforms = compose_transforms(d_config)
#     datasets.append(MegaDepthDataset.from_config(d_config, item_transforms))
#
#     if d.CSV_WARP_PATH in d_config:
#         item_transform_warp = [dt.RandomWarp(),
#                                dt.ToPILImage(),
#                                dt.GrayScale(),
#                                dt.Resize((960, 1280)),
#                                dt.RandomCrop((840, 1120)),
#                                dt.Resize((d_config[d.HEIGHT], d_config[d.WIDTH])),
#                                dt.ToTensor()]
#
#         datasets.append(MegaDepthWarpDataset.from_config(d_config, item_transform_warp))