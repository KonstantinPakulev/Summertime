import os
import cv2
import numpy as np
import pandas as pd
from skimage import io
from Net.source.homography import sample_homography

from torch.utils.data import Dataset

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

        choice = np.random.choice(3)
        if choice == 0:
            homo12 = np.asmatrix(self.annotations.iloc[id, 3:].values).astype(np.float).reshape(3, 3)
            homo21 = homo12.I
        elif choice == 1:
            homo12 = np.asmatrix(sample_homography(image1.shape[1::-1]))
            homo21 = homo12.I

            image2 = cv2.warpPerspective(image1, homo12, image1.shape[1::-1])
        else:
            homo21 = np.asmatrix(sample_homography(image2.shape[1::-1]))
            homo12 = homo21.I

            image1 = cv2.warpPerspective(image2, homo21, image2.shape[1::-1])

        item = {IMAGE1: image1, IMAGE2: image2, HOMO12: homo12, HOMO21: homo21}

        if self.include_sources:
            item[S_IMAGE1] = image1.copy()
            item[S_IMAGE2] = image2.copy()

        if self.item_transforms:
            item = self.item_transforms(item)

        return item
