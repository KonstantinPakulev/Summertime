import os
import h5py
import numpy as np
import pandas as pd
from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms

import Net.source.datasets.dataset_utils as du


class MegaDepthDataset(Dataset):

    @staticmethod
    def from_config(dataset_config, item_transforms):
        return MegaDepthDataset(dataset_config[du.DATASET_ROOT],
                                dataset_config[du.SCENE_INFO_ROOT],
                                dataset_config[du.CSV_PATH],
                                transforms.Compose(item_transforms),
                                dataset_config[du.SOURCES])

    def __init__(self, dataset_root, scene_info_root, csv_path, item_transforms=None, sources=False):
        self.dataset_root = dataset_root
        self.scene_info_root = scene_info_root
        self.annotations = pd.read_csv(csv_path, index_col=[0])
        self.item_transforms = item_transforms
        self.sources = sources

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        iloc = self.annotations.iloc[index]

        scene_name = str(iloc[du.SCENE_NAME])

        id1 = str(iloc[du.ID1])
        id2 = str(iloc[du.ID2])

        image1_name = iloc[du.IMAGE1].split("/")[-1]
        image2_name = iloc[du.IMAGE2].split("/")[-1]

        image1 = io.imread(iloc[du.IMAGE1])
        image2 = io.imread(iloc[du.IMAGE2])

        depth1 = load_depth(iloc[du.DEPTH1])
        depth2 = load_depth(iloc[du.DEPTH2])

        scene_info = np.load(os.path.join(self.scene_info_root, f"{scene_name.zfill(4)}.npz"), allow_pickle=True)

        extrinsics1 = scene_info['poses'][iloc[du.ID1]]
        extrinsics2 = scene_info['poses'][iloc[du.ID2]]

        intrinsics1 = scene_info['intrinsics'][iloc[du.ID1]]
        intrinsics2 = scene_info['intrinsics'][iloc[du.ID2]]

        item = {du.SCENE_NAME: scene_name,
                du.IMAGE1_NAME: image1_name,
                du.IMAGE2_NAME: image2_name,
                du.ID1: id1,
                du.ID2: id2,
                du.IMAGE1: image1, du.IMAGE2: image2,
                du.DEPTH1: depth1, du.DEPTH2: depth2,
                du.EXTRINSICS1: extrinsics1, du.EXTRINSICS2: extrinsics2,
                du.INTRINSICS1: intrinsics1, du.INTRINSICS2: intrinsics2,
                du.SHIFT_SCALE1: np.array([0., 0., 1., 1.]),
                du.SHIFT_SCALE2: np.array([0., 0., 1., 1.])}

        if self.sources:
            item[du.S_IMAGE1] = image1.copy()
            item[du.S_IMAGE2] = image2.copy()

        if self.item_transforms is not None:
            item = self.item_transforms(item)

        return item


"""
Support utils
"""


def load_depth(path):
    with h5py.File(path, 'r') as file:
        data = np.array(file['/depth'])
        return data


# Legacy code

# class MegaDepthWarpDataset(Dataset):
#
#     @staticmethod
#     def from_config(dataset_config, item_transforms):
#         return MegaDepthWarpDataset(dataset_config[du.DATASET_ROOT],
#                                     dataset_config[du.CSV_WARP_PATH],
#                                     transforms.Compose(item_transforms),
#                                     dataset_config[du.SOURCES])
#
#     def __init__(self, dataset_root, csv_path, item_transforms=None, sources=False):
#         self.dataset_root = dataset_root
#         self.annotations = pd.read_csv(csv_path, index_col=[0])
#         self.item_transforms = item_transforms
#         self.sources = sources
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         iloc = self.annotations.iloc[index]
#
#         image1_name = iloc[du.IMAGE1].split("/")[-1]
#         image2_name = image1_name + '_warp'
#
#         image1 = io.imread(iloc[du.IMAGE1])
#
#         item = {du.SCENE_NAME: iloc[du.SCENE_NAME],
#                 du.IMAGE1_NAME: image1_name,
#                 du.IMAGE2_NAME: image2_name,
#                 du.IMAGE1: image1}
#
#         if self.sources:
#             item[du.S_IMAGE1] = image1.copy()
#
#         if self.item_transforms is not None:
#             item = self.item_transforms(item)
#
#         return item
