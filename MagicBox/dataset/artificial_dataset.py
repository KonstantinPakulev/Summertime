import numpy as np
import cv2
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil
import yaml
import tempfile
from skimage import io, transform

from MagicBox.dataset.dataset_pipeline import *

from torch.utils.data import Dataset

primitives_to_draw = [
    'draw_lines',
    'draw_polygon',
    'draw_multiple_polygons',
    'draw_ellipses',
    'draw_star',
    'draw_checkerboard',
    'draw_stripes',
    'draw_cube',
    'gaussian_noise'
]

available_modes = ['training', 'validation', 'test']


class ArtificialDataset(Dataset):

    def __init__(self, mode, config):

        assert mode in available_modes

        self.mode = mode
        self.config = config

        primitives = parse_primitives(config['primitives'], primitives_to_draw)

        base_path = Path(config['data_path'], 'synthetic_shapes' +
                         ('_{}'.format(config['suffix']) if config['suffix'] is not None else ''))
        base_path.mkdir(parents=True, exist_ok=True)

        self.images = []
        self.points = []

        for primitive in primitives:
            tar_path = Path(base_path, '{}.tag.gz'.format(primitive))
            if not tar_path.exists():
                save_primitive_data(primitive, tar_path, config)

            tar = tarfile.open(tar_path)
            temp_dir = Path(tempfile.gettempdir())
            tar.extractall(path=temp_dir)
            tar.close()

            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)

            e = [str(p) for p in Path(path, 'images', self.mode).iterdir()]
            f = [p.replace('images', 'points') for p in e]
            f = [p.replace('.png', '.npy') for p in f]

            self.images.extend(e[:int(truncate * len(e))])
            self.points.extend(f[:int(truncate * len(f))])

    def __len__(self):
        if self.mode == available_modes[0]:
            return len(self.images)
        elif self.mode == available_modes[1]:
            return np.min(len(self.images), self.config['validation_size'])
        elif self.mode == available_modes[2]:
            return np.min(len(self.images), self.config['test_size'])
        else:
            return self.images

    def __getitem__(self, item):
        image_path = self.images[item]
        points_path = self.points[item]

        image = np.expand_dims(io.imread(image_path), axis=0)
        points = np.load(points_path).astype(np.float32)

        # Apply data augmentation
        if self.mode == 'training':
            if self.config['augmentation']['photometric']['enable']:
                image, points =  photometric_augmentation(image, points, self.config['augmentation']['photometric'])
            if self.config['augmentation']['homographic']['enable']:
                image, points = homographic_augmentation(image, points, self.config['augmentation']['homographic'])

        # Convert points to keypoint map
        keypoint_map = get_keypoint_map(image, points)

        return image, keypoint_map