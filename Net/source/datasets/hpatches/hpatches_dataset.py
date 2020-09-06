import os
import numpy as np
import pandas as pd
from skimage import io

from torchvision import transforms

from torch.utils.data import Dataset

import Net.source.datasets.dataset_utils as du


class HPatchesDataset(Dataset):

    @staticmethod
    def from_config(dataset_config, item_transforms):
        return HPatchesDataset(dataset_config[du.DATASET_ROOT],
                               dataset_config[du.CSV_PATH],
                               transforms.Compose(item_transforms),
                               dataset_config[du.SOURCES])

    def __init__(self, dataset_root, csv_path, item_transforms=None, sources=False):
        """
        :param dataset_root: Path to the dataset folder
        :param csv_path: Path to csv file with annotations
        :param item_transforms: Transforms for both homography and image
        :param sources: Provide initial images
        """
        self.dataset_root = dataset_root
        self.annotations = pd.read_csv(csv_path)
        self.item_transforms = item_transforms
        self.sources = sources

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, id):
        iloc = self.annotations.iloc[id]

        folder = iloc[0]
        image1_name = iloc[1]
        image2_name = iloc[2]

        image1_path = os.path.join(self.dataset_root, folder, image1_name)
        image2_path = os.path.join(self.dataset_root, folder, image2_name)

        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)

        homo12 = np.asmatrix(iloc[3:].values).astype(np.float).reshape(3, 3)
        homo21 = homo12.I

        item = {du.SCENE_NAME: folder,
                du.IMAGE1_NAME: folder + "_" + image1_name,
                du.IMAGE2_NAME: folder + "_" + image2_name,
                du.IMAGE1: image1, du.IMAGE2: image2,
                du.H12: homo12, du.H21: homo21}

        if self.sources:
            item[du.S_IMAGE1] = image1.copy()
            item[du.S_IMAGE2] = image2.copy()

        if self.item_transforms:
            item = self.item_transforms(item)

        return item
