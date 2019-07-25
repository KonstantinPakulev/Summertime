import os
import sys
from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events

from Net.exp_env.exp_config import cfg
from Net.nn.model import Net
from Net.nn.criterion import HomoHingeLoss
from Net.hpatches_dataset import (
    HPatchesDataset,
    TRAIN,
    Grayscale,
    Normalize,
    Rescale,
    ToTensor
)
from Net.utils.eval_utils import (nearest_neighbor_match_score,
                                  nearest_neighbor_thresh_match_score,
                                  nearest_neighbor_ratio_match_score)
from Net.utils.ignite_utils import AverageMetric
from Net.utils.image_utils import warp_image, erode_mask, dilate_mask


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

        dot = torch.zeros((1, 1, 240, 320))
        dot[0, 0, 120, 160] = 1
        d_dot = dilate_mask(dot)
        neg = d_dot - dot

        ones = torch.ones((1, 1, 240, 320))
        warped = warp_image(ones, homo).gt(0).float()
        w_eroded = erode_mask(warped)

        all = torch.cat((neg, warped, w_eroded, warped - w_eroded), dim=0)

        print(neg.nonzero().shape)

        writer.add_image("image", make_grid(all))

        return None

    trainer = Engine(iteration)

    writer.close()

    trainer.run(loader, max_epochs=cfg.TRAIN.NUM_EPOCHS)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str)

    args = parser.parse_args()

    _device = torch.device('cpu')
    _log_dir = args.log_dir

    test(_device, _log_dir)
