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
from ignite.engine import Engine

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

from Net.utils.eval_utils import nearest_neighbor_match_score
from Net.utils.image_utils import warp_image, erode_mask, space_to_depth


def test(device, log_dir):

    dataset = HPatchesDataset(root_path=cfg.DATASET.view.root,
                              csv_file=cfg.DATASET.view.csv,
                              mode=TRAIN,
                              split_ratios=[0.01, 0, 0],
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

    writer = SummaryWriter(logdir=log_dir)

    def iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )

        ones = torch.ones_like(im2)
        w_ones = warp_image(ones, homo).gt(0).float()

        morphed_ones = erode_mask(w_ones)

        r_mask = space_to_depth(morphed_ones, cfg.MODEL.GRID_SIZE).prod(dim=1).unsqueeze(0)

        print(r_mask.shape, r_mask.unique())

        # images = torch.cat([ones, w_ones, morphed_ones, w_ones - morphed_ones], dim=0)
        writer.add_image("image", make_grid(r_mask))

        return None

    engine = Engine(iteration)
    engine.run(loader)

    writer.close()


if __name__ == "__main__":
    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    t1 = torch.ones((3, 16, 30, 40))
    t2 = torch.ones((3, 16, 30, 40))
    mask = torch.ones((3, 1, 30, 40))

    print(nearest_neighbor_match_score(t1, t2, mask))

    # test(_device, ".")
