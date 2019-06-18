import matplotlib.pyplot as plt
import os
import yaml
import numpy as np

import torch


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def collate(batch):
    print(batch)
    x, y = batch
    x = x.float()
    return x, y


def get_checkpoint_name(model_name, iter):
    return model_name + '_{}'.format(iter)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint['model'], checkpoint['optimizer']


def save_checkpoint(model, optimizer, path):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path)


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
