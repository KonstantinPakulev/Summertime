import pathlib
import numpy as np
from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms

import Net.source.datasets.dataset_utils as du


class AachenDataset(Dataset):

    @staticmethod
    def from_config(dataset_config, item_transforms):
        return AachenDataset(dataset_config[du.DATASET_ROOT],
                             transforms.Compose(item_transforms),
                             dataset_config[du.SOURCES])

    def __init__(self, dataset_root, item_transforms=None, sources=False):
        self.dataset_root = pathlib.Path(dataset_root)
        self.item_transforms = item_transforms
        self.sources = sources

        self.image_paths = list(self.dataset_root.glob("**/*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image1_path = self.image_paths[index]

        scene_name = str(image1_path.relative_to(self.dataset_root).parent)
        image1_name = image1_path.name

        image1 = io.imread(image1_path)

        item = {
            du.SCENE_NAME: scene_name,
            du.IMAGE1_NAME: image1_name,
            du.IMAGE1: image1,
            du.SHIFT_SCALE1: np.array([0., 0., 1., 1.])
        }

        if self.sources:
            item[du.S_IMAGE1] = image1.copy()

        if self.item_transforms is not None:
            item = self.item_transforms(item)

        return item
