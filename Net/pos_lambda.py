import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events

from Net.config import cfg
from Net.hpatches_dataset import (
    HPatchesDataset,
    TRAIN,
    Grayscale,
    Normalize,
    RandomCrop,
    Rescale,
    ToTensor
)
from Net.utils.ignite_utils import AverageMetric
from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid, warp_image

"""
Calculate pos_lambda for descriptor learning from dataset. 
Let "total" is number of all possible matches between two H/8 x W/8 grids. 
Let "pos" is the number of positive matches between two grids. 
Let "neg is the number of negative matches between two grids.
Since there are more negative matches than positive, we want to balance them in order to train better descriptors.
We need to find such x that pos/total * x = neg / total. 
x = neg / pos.
We calculate ratio x among all examples in TRAIN dataset and average it.
"""


def calculate_ratio(des_size, homo):
    n, _, h, w = des_size

    grid = create_coordinates_grid(des_size) * cfg.MODEL.GRID_SIZE + cfg.MODEL.GRID_SIZE // 2
    grid = grid.type_as(homo).to(homo.device)
    w_grid = warp_coordinates_grid(grid, homo)

    grid = grid.unsqueeze(dim=3).unsqueeze(dim=3)
    w_grid = w_grid.unsqueeze(dim=1).unsqueeze(dim=1)

    grid_dist = torch.norm(grid - w_grid, dim=-1)
    ones = torch.ones_like(grid_dist)
    zeros = torch.zeros_like(grid_dist)
    # Mask with homography induced correspondences
    s = torch.where(grid_dist <= cfg.MODEL.GRID_SIZE - 0.5, ones, zeros)

    pos = s.sum()
    total = n * h ** 2 * w ** 2
    neg = total - pos

    return neg / pos


def calculate_pos_lambda(device, log_dir):

    dataset = HPatchesDataset(root_path=cfg.DATASET.view.root,
                              csv_file=cfg.DATASET.view.csv,
                              mode=TRAIN,
                              split_ratios=cfg.DATASET.SPLIT,
                              transform=transforms.Compose([
                                  Grayscale(),
                                  Normalize(mean=cfg.DATASET.view.MEAN, std=cfg.DATASET.view.STD),
                                  Rescale((960, 1280)),
                                  RandomCrop((720, 960)),
                                  Rescale((240, 320)),
                                  ToTensor(),
                              ]))

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False)

    # writer = SummaryWriter(logdir=log_dir)

    def iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )

        # images = torch.cat([im1, im2, warp_image(im2, homo), warp_image(im1, homo.inverse())], dim=0)
        # writer.add_image("image", make_grid(images))

        des_size = im2.size(0), cfg.MODEL.DESCRIPTOR_SIZE, im2.size(2) / 8, im2.size(3) / 8

        ratio = calculate_ratio(des_size, homo) + calculate_ratio(des_size, homo.inverse())
        ratio /= 2

        x = {'ratio': ratio}

        return x

    engine = Engine(iteration)

    AverageMetric(lambda x: x['ratio']).attach(engine, 'ratio')

    @engine.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        print(engine.state.metrics['ratio'])

    engine.run(loader)

    # writer.close()


if __name__ == "__main__":
    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    calculate_pos_lambda(_device, ".")
