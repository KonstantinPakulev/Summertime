import numpy as np
import cv2
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil

from .dataset_utils import parse_primitives, save_primitive_data
from MagicBox.settings import DATA_PATH

from torch.utils.data import Dataset

default_config = {
    'primitives': 'all',
    'truncate': {},
    'validation_size': -1,
    'test_size': -1,
    'cache_in_memory': False,
    'suffix': None,
    'add_augmentation_to_test_set': False,
    'num_workers': 10,
    'generation': {
        'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
        'image_size': [960, 1280],
        'random_seed': 0,
        'params': {
            'generate_background': {
                'min_kernel_size': 150, 'max_kernel_size': 500,
                'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
            'draw_stripes': {'transform_params': (0.1, 0.1)},
            'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
        },
    },
    'preprocessing': {
        'resize': [240, 320],
        'blur_size': 11,
    },
    'augmentation': {
        'photometric': {
            'enable': False,
            'primitives': 'all',
            'params': {},
            'random_order': True,
        },
        'homographic': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }
}

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

    def __init__(self, mode, **config):

        assert mode in available_modes

        primitives = parse_primitives(config['primitives'], primitives_to_draw)

        base_path = Path(DATA_PATH, 'synthetic_shapes' +
                         ('_{}'.format(config['suffix']) if config['suffix'] is not None else ''))
        base_path.mkdir(parents=True, exist_ok=True)

        self.images = []
        self.points = []

        for primitive in primitives:
            tar_path = Path(base_path, '{}.tag.gz'.format(primitive))
            if not tar_path.exists():
                save_primitive_data(primitive, tar_path, config)

            tar = tarfile.open(tar_path)
            temp_dir = Path(os.environ['TMPDIR'])
            tar.extractall(path=temp_dir)
            tar.close()

            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)

            e = [str(p) for p in Path(path, 'images', mode).iterdir()]
            f = [p.replace('images', 'points') for p in e]
            f = [p.replace('.png', '.npy') for p in f]

            self.images.extend(e[:int(truncate * len(e))])
            self.points.extend(f[:int(truncate * len(f))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # TODO important step. Do not forget data augmentation
        # 1) read image and point coordinates

        pass
