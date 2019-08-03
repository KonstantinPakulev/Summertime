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
from Net.nn.model import NetVGG
from Net.hpatches_dataset import (
    HPatchesDataset,

    IMAGE1,
    IMAGE2,
    HOMO12,
    S_IMAGE1,
    S_IMAGE2,

    Grayscale,
    Normalize,
    Rescale,
    ToTensor
)
from Net.utils.image_utils import warp_image, select_keypoints


def torch2cv(img):
    """
    :type img: 1 x C x H x W
    """
    img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)
    return img


def kp2cv(kp):
    """
    :type kp: K x 4
    """
    return cv2.KeyPoint(kp[3], kp[2], 0)


def test(device, log_dir, checkpoint_path):
    dataset = HPatchesDataset(root_path=cfg.DATASET.view.root,
                              csv_file=cfg.DATASET.view.analyze_csv,
                              transform=transforms.Compose([
                                  Grayscale(),
                                  Normalize(mean=cfg.DATASET.view.MEAN, std=cfg.DATASET.view.STD),
                                  Rescale((960, 1280)),
                                  Rescale((480, 640)),
                                  ToTensor(),
                              ]), include_sources=True)

    loader = DataLoader(dataset, 1, False)

    writer = SummaryWriter(logdir=log_dir)

    model = NetVGG().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    def iteration(engine, batch):
        image1, image2, homo, s_image1, s_image2 = (
            batch[IMAGE1].to(device),
            batch[IMAGE2].to(device),
            batch[HOMO12].to(device),
            batch[S_IMAGE1].to(device),
            batch[S_IMAGE2].to(device)
        )
        homo_inv = homo.inverse()

        score1 = model(image1)
        score2 = model(image2)

        """
        Select keypoints and plot them on images
        """

        top_score1, kp1 = select_keypoints(score1, cfg.LOSS.NMS_THRESH, cfg.LOSS.NMS_K_SIZE, cfg.LOSS.TOP_K)
        top_score2, kp2 = select_keypoints(score2, cfg.LOSS.NMS_THRESH, cfg.LOSS.NMS_K_SIZE, cfg.LOSS.TOP_K)

        s_image1 = torch2cv(s_image1)
        s_image2 = torch2cv(s_image2)

        kp1 = kp1.cpu().detach().numpy()
        kp2 = kp2.cpu().detach().numpy()

        kp1 = list(map(kp2cv, kp1))
        kp2 = list(map(kp2cv, kp2))

        s_image1_kp = cv2.drawKeypoints(s_image1, kp1, None, color=(0, 255, 0))
        s_image2_kp = cv2.drawKeypoints(s_image2, kp2, None, color=(0, 255, 0))

        s_image1_kp = s_image1_kp.transpose((2, 0, 1))
        s_image1_kp = torch.from_numpy(s_image1_kp).unsqueeze(0)

        s_image2_kp = s_image2_kp.transpose((2, 0, 1))
        s_image2_kp = torch.from_numpy(s_image2_kp).unsqueeze(0)

        writer.add_image("images", make_grid(torch.cat((s_image1_kp, s_image2_kp), dim=0)), engine.state.iteration)

        """
        Plot score distributions
        """

        score1 -= score1.min()
        score2 -= score2.min()

        score1 *= 190
        score2 *= 190

        w_score1 = warp_image(score1, homo)
        w_score2 = warp_image(score2, homo_inv)

        print(score1.unique(), score2.unique(), score1.unique().shape, score2.unique().shape)
        print(w_score1.unique(), w_score2.unique(), w_score1.unique().shape, w_score2.unique().shape)

        writer.add_image("scores", make_grid(torch.cat((score1, score2), dim=0)), engine.state.iteration)
        writer.add_image("warped_scores", make_grid(torch.cat((w_score1, w_score2), dim=0)), engine.state.iteration)

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

    test(_device, _log_dir, _checkpoint_path)
