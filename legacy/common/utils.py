import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import numpy as np
import os

import torch

# Dataset item dictionary keys
IMAGE = 'image'
WARPED_IMAGE = 'warped_image'

NAME = 'name'
POINTS = 'points'
DEPTH = 'depth'
HOMOGRAPHY = 'homography'

KEYPOINT_MAP = 'keypoint_map'
WARPED_KEYPOINT_MAP = 'warped_keypoint_map'

MASK = 'mask'
WARPED_MASK = 'warped_mask'

# Modes of the dataset
TRAINING = 'training'
VALIDATION = 'validation'
TEST = 'test'


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_checkpoint_path(experiment_config, model_config, epoch):
    base_path = Path(experiment_config['checkpoints_path'])
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path.joinpath(model_config['name'] + '_{}.torch'.format(epoch))


def clear_old_checkpoints(experiment_config):
    base_path = os.path.join(experiment_config['checkpoints_path'])
    if os.path.exists(base_path):
        checkpoints = sorted([os.path.join(base_path, file) for file in os.listdir(base_path)], key=os.path.getmtime)
        for cp in checkpoints[:-experiment_config['keep_checkpoints']]:
            os.remove(cp)


def load_checkpoint(path, map_location=None):
    checkpoint = torch.load(path, map_location)
    return checkpoint['epoch'], checkpoint['model'], checkpoint['optimizer']


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path)


def get_logs_path(experiment_config):
    base_path = Path(experiment_config['logs_path'])
    base_path.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(base_path):
        os.remove(os.path.join(base_path, file))

    return base_path


def plot_images(images, titles=None, cmap='brg', ylabel='', normalize=False, axes=None, dpi=100):
    n = len(images)
    if not isinstance(cmap, list):
        cmap = [cmap] * n
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(6 * n, 6), dpi=dpi)
        if n == 1:
            axes = [axes]
    else:
        if not isinstance(axes, list):
            axes = [axes]
        assert len(axes) == len(images)
    for i in range(n):
        if images[i].shape[-1] == 3:
            images[i] = images[i][..., ::-1]  # BGR to RGB
        axes[i].imshow(images[i], cmap=plt.get_cmap(cmap[i]),
                       vmin=None if normalize else 0,
                       vmax=None if normalize else 1)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():  # remove frame
            spine.set_visible(False)
    axes[0].set_ylabel(ylabel)
    plt.tight_layout()


def grayscale2rgb(image):
    assert len(image.shape) == 2
    return np.stack((image,) * 3, axis=0)


def rgb2grayscale(image):
    assert image.shape[0] == 3
    return image[0]


def normalize_image(image):
    assert image.ravel().max() <= 255 and image.dtype == np.uint8
    image = image.astype(np.float) / 255.0
    return image


def to255scale(image):
    assert image.ravel().max() <= 1
    return (image * 255).astype(np.uint8)


def read_tum_list(base_path, filename):
    tum_list = []

    with open(os.path.join(base_path, filename), 'r') as f:
        lines = f.readlines()[3:]
        for line in lines:
            path = line.rstrip().split()[1]
            tum_list.append(path)

    return tum_list
