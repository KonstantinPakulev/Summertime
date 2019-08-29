import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from ignite.engine import Engine, Events

from legacy.exp_env.analyze_config import cfg as cfg_exp
from Net.source.hpatches_dataset_old import (
    HPatchesDatasetOld,
    GrayscaleOld,
    NormalizeOld,
    RandomCropOld,
    RescaleOld,
    ToTensorOld
)
from Net.utils.ignite_utils import AverageMetric
from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid, dilate_filter


def calculate_ratio(des_size, homo, cfg):
    n, _, h, w = des_size

    grid = create_coordinates_grid(des_size) * cfg.MODEL.GRID_SIZE + cfg.MODEL.GRID_SIZE // 2
    grid = grid.type_as(homo).to(homo.device)
    w_grid = warp_coordinates_grid(grid, homo)

    grid = grid.unsqueeze(dim=3).unsqueeze(dim=3)
    w_grid = w_grid.unsqueeze(dim=1).unsqueeze(dim=1)

    # Mask with homography induced correspondences
    grid_dist = torch.norm(grid - w_grid, dim=-1)
    ones = torch.ones_like(grid_dist)
    zeros = torch.zeros_like(grid_dist)
    s = torch.where(grid_dist <= cfg.MODEL.GRID_SIZE - 0.5, ones, zeros)

    # Create negative correspondences around positives
    ns = s.clone().view(-1, 1, h, w)
    ns = dilate_filter(ns).view(n, h, w, h, w)
    ns = ns - s

    pos = s.sum()
    neg = ns.sum()

    return neg / pos


def calculate_pos_lambda(device, cfg):
    dataset = HPatchesDatasetOld(root_path=cfg.DATASET.view.root,
                                 csv_file=cfg.DATASET.view.csv,
                                 transform=transforms.Compose([
                                  GrayscaleOld(),
                                  NormalizeOld(mean=cfg.DATASET.view.MEAN, std=cfg.DATASET.view.STD),
                                  RescaleOld((960, 1280)),
                                  RandomCropOld((720, 960)),
                                  RescaleOld((240, 320)),
                                  ToTensorOld(),
                              ]))

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True)

    def iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )

        des_size = im2.size(0), cfg.MODEL.DESCRIPTOR_SIZE, im2.size(2) // 8, im2.size(3) // 8

        ratio = calculate_ratio(des_size, homo, cfg) + calculate_ratio(des_size, homo.inverse(), cfg)
        ratio /= 2

        x = {'ratio': ratio}

        return x

    engine = Engine(iteration)

    AverageMetric(lambda x: x['ratio']).attach(engine, 'ratio')

    @engine.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        print(engine.state.metrics['ratio'])

    engine.run(loader)


if __name__ == "__main__":
    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    calculate_pos_lambda(_device, cfg_exp)
