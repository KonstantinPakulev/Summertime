import os
import sys
import datetime
from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers.param_scheduler import LRScheduler

from Net.config import cfg
from Net.nn.model import Net
from Net.nn.criterion import HomoHingeLoss
from Net.hpatches_dataset import (
    HPatchesDataset,
    TRAIN,
    VALIDATE,
    Grayscale,
    Normalize,
    RandomCrop,
    Rescale,
    ToTensor
)
from Net.utils.common_utils import print_dict
from Net.utils.eval_utils import nearest_neighbor_match_score
from Net.utils.ignite_utils import AverageMetric
from Net.utils.image_utils import warp_image, erode_mask


# TODO. The mask should be produced by detector in future. LATER
def create_bordering_mask(im2, homo):
    mask = torch.ones_like(im2).to(homo.device)
    w_mask = warp_image(mask, homo).gt(0).float()

    morphed_mask = erode_mask(w_mask)

    return morphed_mask


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

    train_dataset = HPatchesDataset(root_path=cfg.DATASET.TRAIN.root,
                                    csv_file=cfg.DATASET.TRAIN.csv,
                                    mode=TRAIN,
                                    split_ratios=cfg.DATASET.SPLIT,
                                    transform=transforms.Compose([
                                        Grayscale(),
                                        Normalize(mean=cfg.DATASET.TRAIN.MEAN, std=cfg.DATASET.TRAIN.STD),
                                        Rescale((960, 1280)),
                                        RandomCrop((720, 960)),
                                        Rescale((240, 320)),
                                        ToTensor(),
                                    ]))

    val_dataset = HPatchesDataset(root_path=cfg.DATASET.VAL.root,
                                  csv_file=cfg.DATASET.VAL.csv,
                                  mode=VALIDATE,
                                  split_ratios=cfg.DATASET.SPLIT,
                                  transform=transforms.Compose([
                                      Grayscale(),
                                      Normalize(mean=cfg.DATASET.VAL.MEAN, std=cfg.DATASET.VAL.STD),
                                      Rescale((960, 1280)),
                                      Rescale((240, 320)),
                                      ToTensor(),
                                  ]))

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.VAL.BATCH_SIZE,
                            shuffle=True,
                            num_workers=num_workers)

    """
    Model, optimizer and criterion settings. 
    Training and validation steps. 
    """

    model = Net(cfg.MODEL.GRID_SIZE, cfg.MODEL.DESCRIPTOR_SIZE).to(device)
    criterion = HomoHingeLoss(cfg.MODEL.GRID_SIZE, cfg.LOSS.POS_LAMBDA,
                              cfg.LOSS.POS_MARGIN, cfg.LOSS.NEG_MARGIN)
    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.SCH_STEP, gamma=cfg.TRAIN.SCH_GAMMA)

    def train_iteration(engine, batch):
        optimizer.zero_grad()

        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )

        mask = create_bordering_mask(im2, homo)

        raw_desc1, _ = model(im1)
        raw_desc2, _ = model(im2)

        # TODO. #1 Double training.

        loss, s, dot_desc, r_mask = criterion(raw_desc1, raw_desc2, homo, mask)
        loss.backward()

        optimizer.step()

        return {
            'loss': loss,
            's': s,
            'dot_desc': dot_desc,
            'r_mask': r_mask
        }

    def inference_iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'].to(device),
            batch['im2'].to(device),
            batch['homo'].to(device)
        )

        mask = create_bordering_mask(im2, homo)

        raw_desc1, _ = model(im1)
        raw_desc2, _ = model(im2)

        loss, s, dot_desc, r_mask = criterion(raw_desc1, raw_desc2, homo, mask)

        return {
            'loss': loss,
            's': s,
            'dot_desc': dot_desc,
            'r_mask': r_mask
        }

    trainer = Engine(train_iteration)
    tester = Engine(inference_iteration)

    trainer.add_event_handler(Events.ITERATION_STARTED, LRScheduler(lr_scheduler))

    """
    Visualisation utils, logging and metrics
    """

    writer = SummaryWriter(logdir=log_dir)

    # TODO. Save the best model by providing score function and include it in files name. LATER
    # TODO. Add more metric functions. LATER

    """
    Metric functions
    """

    def l_loss(x):
        return x['loss']

    def l_nn_match(x):
        return nearest_neighbor_match_score(x['s'], x['dot_desc'], x['r_mask'])

    """
    Metrics for trainer
    """
    AverageMetric(l_loss, cfg.TRAIN.LOG_INTERVAL).attach(trainer, 'loss')
    AverageMetric(l_nn_match, cfg.TRAIN.LOG_INTERVAL).attach(trainer, 'nn_match')

    """
    Metrics for tester
    """
    AverageMetric(l_loss).attach(tester, 'loss')
    AverageMetric(l_nn_match).attach(tester, 'nn_match')

    """
    Registering callbacks for sending summary to tensorboard
    """
    epoch_timer = Timer(average=False)
    batch_timer = Timer(average=True)

    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iteration_completed(engine):
        if engine.state.iteration % cfg.TRAIN.LOG_INTERVAL == 0:
            writer.add_scalar("train/loss", engine.state.metrics['loss'], engine.state.iteration)
            writer.add_scalar("train/nn_match_score", engine.state.metrics['nn_match'], engine.state.iteration)

        if engine.state.iteration % cfg.VAL.LOG_INTERVAL == 0:
            tester.run(val_loader)

            writer.add_scalar("val/loss", tester.state.metrics['loss'], engine.state.iteration)
            writer.add_scalar("val/nn_match_score", tester.state.metrics['nn_match'], engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        tester.run(val_loader)

        text = f"""
                Epoch {engine.state.epoch} completed.
                \tFinished in {datetime.timedelta(seconds=epoch_timer.value())}.
                \tAverage time per batch is {batch_timer.value():.2f} seconds
                \tValidation loss is {tester.state.metrics['loss']: .4f}
                \tNN match score is: {tester.state.metrics['nn_match']: .4f}
                \tLearning rate is: {optimizer.param_groups[0]["lr"]}
                """

        # TODO. #2 show matches and detections each n epochs. Images

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