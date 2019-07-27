import os
import sys
import datetime
from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer

from Net.config import cfg
from Net.nn.model import Net
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

from Net.utils.common_utils import print_dict
from Net.utils.eval_utils import (l_loss,
                                  l_det_loss,
                                  l_des_loss,
                                  l_collect_show,
                                  plot_keypoints)

from Net.utils.ignite_utils import AverageMetric, CollectMetric


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


def attach_metrics(engine):
    AverageMetric(l_loss, cfg.TRAIN.LOG_INTERVAL).attach(engine, 'loss')
    AverageMetric(l_det_loss, cfg.TRAIN.LOG_INTERVAL).attach(engine, 'det_loss')
    AverageMetric(l_des_loss, cfg.TRAIN.LOG_INTERVAL).attach(engine, 'des_loss')


def output_metrics(writer, data_engine, state_engine, tag):
    """
    :param writer: SummaryWriter
    :param data_engine: Engine to take data from
    :param state_engine: Engine to take current state from
    :param tag: Category to write data to
    """
    writer.add_scalar(f"{tag}/loss", data_engine.state.metrics['loss'], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/det_loss", data_engine.state.metrics['det_loss'], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/des_loss", data_engine.state.metrics['des_loss'], state_engine.state.iteration)


def train(device, num_workers, log_dir, checkpoint_dir):
    """
    :param device: cpu or gpu
    :param num_workers: number of workers to load the data
    :param log_dir: path to the directory to store tensorboard db
    :param checkpoint_dir: path to the directory to save checkpoints
    """
    """
    Dataset and data loaders preparation.
    """

    train_loader = DataLoader(h_patches_dataset(TRAIN),
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(h_patches_dataset(VALIDATE),
                            batch_size=cfg.VAL.BATCH_SIZE,
                            shuffle=True,
                            num_workers=num_workers)

    val_show_loader = DataLoader(h_patches_dataset(VALIDATE_SHOW),
                                 batch_size=cfg.VAL_SHOW.BATCH_SIZE)

    """
    Model, optimizer and criterion settings. 
    Training and validation steps. 
    """

    model = Net(cfg.MODEL.GRID_SIZE, cfg.MODEL.DESCRIPTOR_SIZE).to(device)

    det_criterion = HomoMSELoss(cfg.LOSS.NMS_THRESH, cfg.LOSS.NMS_K_SIZE,
                                cfg.LOSS.TOP_K,
                                cfg.LOSS.GAUSS_K_SIZE, cfg.LOSS.GAUSS_SIGMA)
    des_criterion = HomoHingeLoss(cfg.MODEL.GRID_SIZE, cfg.LOSS.POS_LAMBDA,
                                  cfg.LOSS.POS_MARGIN, cfg.LOSS.NEG_MARGIN)

    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    def iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )
        homo_inv = homo.inverse()

        det1, des1 = model(im1)
        det2, des2 = model(im2)

        det_loss1, top_k_mask1, vis_mask1 = det_criterion(det1, det2, homo)
        det_loss2, top_k_mask2, vis_mask2 = det_criterion(det2, det1, homo_inv)

        des_loss1, s1, dot_des1 = des_criterion(des1, des2, homo, vis_mask1)
        des_loss2, s2, dot_des2 = des_criterion(des2, des1, homo_inv, vis_mask2)

        det_loss = (det_loss1 + det_loss2) / 2
        des_loss = cfg.LOSS.DES_LAMBDA * (des_loss1 + des_loss2) / 2

        loss = det_loss + des_loss

        return {
            'loss': loss,
            'det_loss': det_loss,
            'des_loss': des_loss,

            'top_k_mask1': top_k_mask1,
            's1': s1,
            'dot_des1': dot_des1,

            'top_k_mask2': top_k_mask2,
            's2': s2,
            'dot_des2': dot_des2
        }

    def train_iteration(engine, batch):
        optimizer.zero_grad()

        endpoint = iteration(engine, batch)

        endpoint['loss'].backward()

        optimizer.step()

        return {
            'loss': endpoint['loss'],
            'det_loss': endpoint['det_loss'],
            'des_loss': endpoint['des_loss']
        }

    def validation_iteration(engine, batch):
        endpoint = iteration(engine, batch)

        return {
            'loss': endpoint['loss'],
            'det_loss': endpoint['det_loss'],
            'des_loss': endpoint['des_loss']
        }

    def validation_show_iteration(engine, batch):
        endpoint = iteration(engine, batch)

        return {
            'im1': batch['orig1'],
            'im2': batch['orig2'],
            'top_k_mask1': endpoint['top_k_mask1'],
            'top_k_mask2': endpoint['top_k_mask2']
        }

    trainer = Engine(train_iteration)
    validator = Engine(validation_iteration)
    validator_show = Engine(validation_show_iteration)

    # TODO. Save the best model by providing score function and include it in files name. LATER

    """
    Visualisation utils, logging and metrics
    """

    writer = SummaryWriter(logdir=log_dir)

    attach_metrics(trainer)
    attach_metrics(validator)
    CollectMetric(l_collect_show).attach(validator_show, 'show')

    """
    Registering callbacks for sending summary to tensorboard
    """
    epoch_timer = Timer(average=False)
    batch_timer = Timer(average=True)

    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iteration_completed(engine):
        if engine.state.iteration % cfg.TRAIN.LOG_INTERVAL == 0:
            output_metrics(writer, engine, engine, "train")

        if engine.state.iteration % cfg.VAL.LOG_INTERVAL == 0:
            validator.run(val_loader)
            output_metrics(writer, validator, engine, "val")

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        if engine.state.epoch % cfg.VAL_SHOW.LOG_INTERVAL == 0:
            validator_show.run(val_show_loader)
            # TODO. #2 show matches each n epochs. Images
            plot_keypoints(writer, engine.state.epoch, validator_show.state.metrics['show'])

        # TODO. Fancy printing of all other metrics later. REALLY LATER. ALMOST LAST.
        # validator.run(val_loader)
        text = f"""
                Epoch {engine.state.epoch} completed.
                \tFinished in {datetime.timedelta(seconds=epoch_timer.value())}.
                \tAverage time per batch is {batch_timer.value():.2f} seconds
                \tLearning rate is: {optimizer.param_groups[0]["lr"]}
                """
        writer.add_text("Log", text, engine.state.epoch)

    epoch_timer.attach(trainer,
                       start=Events.EPOCH_STARTED,
                       resume=Events.ITERATION_STARTED,
                       pause=Events.ITERATION_COMPLETED)
    batch_timer.attach(trainer,
                       start=Events.EPOCH_STARTED,
                       resume=Events.ITERATION_STARTED,
                       pause=Events.ITERATION_COMPLETED,
                       step=Events.ITERATION_COMPLETED)

    trainer.run(train_loader, max_epochs=cfg.TRAIN.NUM_EPOCHS)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)

    args = parser.parse_args()

    print_dict(cfg)

    _device, _num_workers = (torch.device('cuda'), 8) if torch.cuda.is_available() else (torch.device('cpu'), 0)
    _log_dir = args.log_dir
    _checkpoint_dir = args.checkpoint_dir

    train(_device, _num_workers, _log_dir, _checkpoint_dir)
