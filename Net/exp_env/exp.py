import os
import sys
from skimage import io, color
import numpy as np
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
from legacy.ST_Net.model.st_det_vgg import STDetVGGModule
from legacy.ST_Net.model.st_net_vgg import STNetVGGModule
from Net.nn.criterion import HomoMSELoss, HomoHingeLoss
from Net.hpatches_dataset import (
    HPatchesDataset,
    TRAIN,
    VALIDATE,
    VALIDATE_SHOW,
    Grayscale,
    Normalize,
    RandomCrop,
    Rescale,
    ToTensor
)
from Net.utils.eval_utils import (l_collect_show,
                                  nearest_neighbor_match_score,
                                  nearest_neighbor_thresh_match_score,
                                  nearest_neighbor_ratio_match_score,
                                  plot_keypoints)
from Net.utils.ignite_utils import AverageMetric, CollectMetric
from Net.utils.image_utils import warp_image, erode_filter, dilate_filter, nms


def h_patches_dataset(mode):
    transform = [Grayscale(),
                 Normalize(mean=cfg.DATASET.view.MEAN, std=cfg.DATASET.view.STD),
                 Rescale((960, 1280))]

    if mode == TRAIN:
        csv_file = cfg.DATASET.view.train_csv
        transform += [RandomCrop((720, 960)),
                      Rescale((240, 320))]
        include_originals = False

    elif mode == VALIDATE:
        csv_file = cfg.DATASET.view.val_csv
        transform += [Rescale((240, 320))]
        include_originals = False
    else:
        csv_file = cfg.DATASET.view.val_show_csv
        transform += [Rescale((480, 640))]
        include_originals = True

    transform += [ToTensor()]

    return HPatchesDataset(root_path=cfg.DATASET.view.root,
                           csv_file=csv_file,
                           transform=transforms.Compose(transform),
                           include_originals=include_originals)


def test(device, log_dir):
    dataset = HPatchesDataset(root_path=cfg.DATASET.view.root,
                              csv_file='val.csv',
                              transform=transforms.Compose([
                                  Grayscale(),
                                  Normalize(mean=cfg.DATASET.view.MEAN, std=cfg.DATASET.view.STD),
                                  Rescale((240, 320)),
                                  ToTensor(),
                              ]))

    val_show_loader = DataLoader(h_patches_dataset(VALIDATE_SHOW),
                                 batch_size=cfg.VAL_SHOW.BATCH_SIZE)

    loader = DataLoader(dataset,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=False)

    writer = SummaryWriter(logdir=log_dir)

    model = Net(cfg.MODEL.GRID_SIZE, cfg.MODEL.DESCRIPTOR_SIZE).to(device)
    model_lf = STDetVGGModule(cfg.MODEL.GRID_SIZE,
                              cfg.LOSS.NMS_THRESH, cfg.LOSS.NMS_K_SIZE,
                              cfg.LOSS.TOP_K,
                              cfg.LOSS.GAUSS_K_SIZE, cfg.LOSS.GAUSS_SIGMA)
    lf = STNetVGGModule(model_lf, None)

    criterion = HomoMSELoss(cfg.LOSS.NMS_THRESH, cfg.LOSS.NMS_K_SIZE,
                            cfg.LOSS.TOP_K,
                            cfg.LOSS.GAUSS_K_SIZE, cfg.LOSS.GAUSS_SIGMA)
    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    def iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )
        homo_inv = homo.inverse()

        det1, _ = model(im1)
        det2, _ = model(im2)

        det_loss1, top_k_mask1, vis_mask1 = criterion(det1, det2, homo)
        det_loss2, top_k_mask2, vis_mask2 = criterion(det2, det1, homo_inv)

        det_loss = (det_loss1 + det_loss2) / 2

        # loss, _, _ = criterion(det1, det2, homo)
        # print(loss)
        #
        # im1_score, _, _ = model_lf.process(det1)
        # im1_gtsc, _, _, im1_visible = lf.get_gt_score(det2, homo)
        # loss2 = model_lf.loss(im1_score, im1_gtsc, im1_visible)
        # print(loss2)
        # _, _, gt_2, vm2 = criterion(det2, det1, homo_inv)

        # gt_lf_1, _, _, vm_lf_1 = lf.get_gt_score(det2, homo)
        # gt_lf_2, _, _, vm_lf_2 = lf.get_gt_score(det1, homo_inv)
        #
        # my_images = torch.cat((gt_1, gt_2), dim=0)
        # lf_images = torch.cat((gt_lf_1, gt_lf_2), dim=0)
        #
        # writer.add_image("my", make_grid(my_images))
        # writer.add_image("lf", make_grid(lf_images))

        return det_loss, top_k_mask1, top_k_mask2, vis_mask1, vis_mask2

    def validation_show_iteration(engine, batch):
        _, top_k_mask1, top_k_mask2, vis_mask1, vis_mask2 = iteration(engine, batch)

        return {
            'im1': batch['orig1'],
            'im2': batch['orig2'],
            'top_k_mask1': top_k_mask1,
            'top_k_mask2': top_k_mask2
        }

    trainer = Engine(iteration)
    validator_show = Engine(validation_show_iteration)

    CollectMetric(l_collect_show).attach(validator_show, 'show')

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        validator_show.run(val_show_loader)
        plot_keypoints(writer, engine.state.epoch, validator_show.state.metrics['show'])

    trainer.run(loader, max_epochs=cfg.TRAIN.NUM_EPOCHS)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str)

    args = parser.parse_args()

    _device = torch.device('cpu')
    _log_dir = args.log_dir

    test(_device, _log_dir)
