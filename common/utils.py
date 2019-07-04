import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import numpy as np
import os

import torch
from torch.utils.data._utils.collate import default_collate


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def collate(batch):
    x, y = default_collate(batch)
    return x.float(), y.float()


def get_checkpoint_path(experiment_config, model_config, epoch):
    base_path = Path(experiment_config['checkpoints_path'], experiment_config['name'])
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path.joinpath(model_config['name'] + '_{}.torch'.format(epoch))


def clear_old_checkpoints(experiment_config):
    base_path = os.path.join(experiment_config['checkpoints_path'], experiment_config['name'])
    if os.path.exists(base_path):
        checkpoints = sorted([os.path.join(base_path, file) for file in os.listdir(base_path)], key=os.path.getmtime)
        for cp in checkpoints[:-experiment_config['keep_checkpoints']]:
            os.remove(cp)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint['epoch'], checkpoint['model'], checkpoint['optimizer']


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path)


def init_log_dir(logs_base, experiment_config):
    log_dir = os.path.join(logs_base, experiment_config['name'])
    for file in os.listdir(log_dir):
        os.remove(os.path.join(log_dir, file))
    return log_dir


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
    return np.stack((image,) * 3, axis=0)


def rgb2grayscale(image):
    return image[0]


def rgb2gbr(image):
    return image.transpose((1, 2, 0)).astype(np.uint8)


def normalize_image(image):
    image /= 255
    return image


def to255scale(image):
    return (image * 255).astype(np.uint8)