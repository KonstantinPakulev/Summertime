import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter
from ignite.engine import Engine

from Net.test_env.test_config import cfg
from Net.hpatches_dataset import (
    HPatchesDataset,
    TRAIN,
    Grayscale,
    Normalize,
    Rescale,
    ToTensor
)


def test(device, log_dir):

    dataset = HPatchesDataset(root_path=cfg.DATASET.view.root,
                              csv_file=cfg.DATASET.view.csv,
                              mode=TRAIN,
                              split_ratios=cfg.DATASET.SPLIT,
                              transform=transforms.Compose([
                                  Grayscale(),
                                  Normalize(mean=cfg.DATASET.view.MEAN, std=cfg.DATASET.view.STD),
                                  Rescale((240, 320)),
                                  ToTensor(),
                              ]))

    loader = DataLoader(dataset,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=False)

    writer = SummaryWriter(logdir=log_dir)

    def iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )

        return None

    engine = Engine(iteration)
    engine.run(loader)

    writer.close()


if __name__ == "__main__":
    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test(_device, "./runs")
