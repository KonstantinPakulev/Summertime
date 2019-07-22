import os
import datetime
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer

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
from Net.utils.ignite_utils import AverageMetric
from Net.utils.eval_utils import nearest_neighbor_match_score


def train(device, log_dir, save_dir):
    """
    :param device: cpu or gpu
    :param log_dir: path to the directory to store tensorboard db
    :param save_dir: path to the directory to save checkpoints
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
                                        ToTensor(device),
                                    ]))

    val_dataset = HPatchesDataset(root_path=cfg.DATASET.VAL.root,
                                  csv_file=cfg.DATASET.VAL.csv,
                                  mode=VALIDATE,
                                  split_ratios=cfg.DATASET.SPLIT,
                                  transform=transforms.Compose([
                                      Grayscale(),
                                      Normalize(mean=cfg.DATASET.VAL.MEAN, std=cfg.DATASET.VAL.STD),
                                      Rescale((960, 1280)),
                                      RandomCrop((720, 960)),
                                      Rescale((240, 320)),
                                      ToTensor(device),
                                  ]))

    # TODO. Try different batch size

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              num_workers=8)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.VAL.BATCH_SIZE,
                            shuffle=True,
                            num_workers=8)

    """
    Model, optimizer and criterion settings. 
    Training and validation steps. 
    """

    # TODO. Think of lr scheduler.
    model = Net(cfg.MODEL.GRID_SIZE, cfg.MODEL.DESCRIPTOR_SIZE).to(device)
    # TODO. Find out which lambda is the best
    criterion = HomoHingeLoss(cfg.MODEL.GRID_SIZE, cfg.LOSS.POS_LAMBDA,
                              cfg.LOSS.POS_MARGIN, cfg.LOSS.NEG_MARGIN)
    optimizer = Adam(model.parameters(), weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    def train_iteration(engine, batch):
        optimizer.zero_grad()

        im1, im2, homo = (
            batch['im1'],
            batch['im2'],
            batch['homo']
        )

        raw_desc1, desc1 = model(im1)
        raw_desc2, desc2 = model(im2)

        # TODO. Double training.
        # TODO. Do not forget about masks: both for bordering artifacts and ordinary.

        loss = criterion(raw_desc1, raw_desc2, homo)
        loss.backward()

        optimizer.step()

        # TODO. Purely for mac testing use desc# instead raw_desc#

        return {
            'loss': loss,
            'desc1': F.normalize(raw_desc1),
            'desc2': F.normalize(raw_desc2)
        }

    def inference_iteration(engine, batch):
        im1, im2, homo = (
            batch['im1'],
            batch['im2'],
            batch['homo']
        )

        raw_desc1, desc1 = model(im1)
        raw_desc2, desc2 = model(im2)

        loss = criterion(raw_desc1, raw_desc2, homo)

        # TODO. Purely for mac testing use desc# instead raw_desc#

        return {
            'loss': loss,
            'desc1': F.normalize(raw_desc1),
            'desc2': F.normalize(raw_desc2)
        }

    """
    Visualisation utils, logging and metrics
    """

    writer = SummaryWriter(logdir=log_dir)

    trainer = Engine(train_iteration)
    tester = Engine(inference_iteration)

    # TODO. Save the best model by providing score function and include it in files name

    epoch_timer = Timer(average=False)
    batch_timer = Timer(average=True)

    # TODO. Add more metric functions.

    """
    Metric functions
    """

    def l_loss(x):
        return x['loss']

    def l_nn_match(x):
        return nearest_neighbor_match_score(x['desc1'], x['desc2'])

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

    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iteration_completed(engine):
        if engine.state.iteration % cfg.TRAIN.LOG_INTERVAL == 0:
            tester.run(val_loader)

            writer.add_scalar("train/loss", engine.state.metrics['loss'], engine.state.iteration)
            writer.add_scalar("train/nn_match_score", engine.state.metrics['nn_match'], engine.state.iteration)

            writer.add_scalar("val/loss", tester.state.metrics['loss'], engine.state.iteration)
            writer.add_scalar("val/nn_match_score", tester.state.metrics['nn_match'], engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        tester.run(val_loader)

        text = f"""
                Epoch {engine.state.epoch} completed.
                \tFinished in {datetime.timedelta(seconds=epoch_timer.value())}.
                \tAverage time per batch is {batch_timer.value():.2f} seconds
                \tValidation loss is {tester.state.metrics['loss']:.4f}
                \tNN match score is: {tester.state.metrics['nn_match']:.4f}
                """

        # TODO. show matches and detections each n epochs. Images

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

    args = parser.parse_args()

    # TODO. Probably rewrite cfg to be a yaml file or smth similar
    #  Because we may need to use different configs to launch
    #  Accept path to config in args and pass it to train

    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _log_dir = os.path.join('./runs/', datetime.datetime.now().strftime("%H.%M.%S_%d.%m.%Y"))
    if not os.path.exists(_log_dir):
        os.mkdir(_log_dir)

    train(_device, _log_dir, None)
