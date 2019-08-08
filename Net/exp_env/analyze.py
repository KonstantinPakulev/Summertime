import os
import cv2
import sys
import numpy as np
from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events

from Net.exp_env.analyze_config import cfg
from Net.nn.model import Net
from Net.nn.criterion import HardTripletLoss
from Net.hpatches_dataset import (
    HPatchesDataset,

    TRAIN,
    VALIDATE,
    VALIDATE_SHOW,

    IMAGE1,
    IMAGE2,
    HOMO12,
    HOMO21,
    S_IMAGE1,
    S_IMAGE2,

    Grayscale,
    Normalize,
    RandomCrop,
    Rescale,
    ToTensor
)
from Net.utils.model_utils import sample_descriptors
from Net.utils.image_utils import warp_image, select_keypoints, warp_keypoints


def h_patches_dataset(mode, config):
    transform = [Grayscale(),
                 Normalize(mean=config.DATASET.view.MEAN, std=config.DATASET.view.STD),
                 Rescale((960, 1280))]

    if mode == TRAIN:
        csv_file = config.DATASET.view.train_csv
        transform += [RandomCrop((720, 960)),
                      Rescale((240, 320))]
        include_originals = False

    elif mode == VALIDATE:
        csv_file = config.DATASET.view.val_csv
        transform += [Rescale((240, 320))]
        include_originals = False
    else:
        csv_file = config.DATASET.view.val_show_csv
        transform += [Rescale((320, 640))]
        include_originals = True

    transform += [ToTensor()]

    return HPatchesDataset(root_path=config.DATASET.view.root,
                           csv_file=csv_file,
                           transform=transforms.Compose(transform),
                           include_sources=include_originals)


def test(config, device, log_dir, checkpoint_path):
    loader = DataLoader(h_patches_dataset(TRAIN, config),
                        batch_size=config.TRAIN.BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)

    writer = SummaryWriter(logdir=log_dir)

    model = Net(config.MODEL.DESCRIPTOR_SIZE).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    triplet_criterion = HardTripletLoss(config.MODEL.GRID_SIZE, config.LOSS.MARGIN, config.LOSS.DES_LAMBDA_TRI)

    def iteration(engine, batch):
        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(device),
            batch[IMAGE2].to(device),
            batch[HOMO12].to(device),
            batch[HOMO21].to(device)
        )

        score1, desc1 = model(image1)
        score2, desc2 = model(image2)

        _, kp1 = select_keypoints(score1, config.LOSS.NMS_THRESH, config.LOSS.NMS_K_SIZE, config.LOSS.TOP_K)
        _, kp2 = select_keypoints(score2, config.LOSS.NMS_THRESH, config.LOSS.NMS_K_SIZE, config.LOSS.TOP_K)

        kp1_desc = sample_descriptors(desc1, kp1, config.MODEL.GRID_SIZE)
        kp2_desc = sample_descriptors(desc2, kp2, config.MODEL.GRID_SIZE)

        torch.save(kp1, 'kp1.torch')
        torch.save(kp2, 'kp2.torch')

        torch.save(desc1, 'desc1.torch')
        torch.save(desc2, 'desc2.torch')

        w_kp1 = warp_keypoints(kp1, homo12)
        w_kp2 = warp_keypoints(kp2, homo21)

        des_loss1 = triplet_criterion(kp1, w_kp1, kp1_desc, desc2)

        return 0

    validator_show = Engine(iteration)

    validator_show.run(loader)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_path", type=str)

    args = parser.parse_args()

    _device = torch.device('cpu')
    _log_dir = args.log_dir
    _checkpoint_path = args.checkpoint_path

    torch.manual_seed(9)

    test(cfg, _device, _log_dir, _checkpoint_path)
