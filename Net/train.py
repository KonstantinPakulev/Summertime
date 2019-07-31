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
from Net.nn.model import NetVGG, NetRF, Net
from Net.nn.criterion import HomoMSELoss, HomoHingeLoss
from Net.hpatches_dataset import (
    HPatchesDataset,

    TRAIN,
    VALIDATE,
    VALIDATE_SHOW,

    IMAGE1,
    IMAGE2,
    HOMO,
    S_IMAGE1,
    S_IMAGE2,

    Grayscale,
    Normalize,
    RandomCrop,
    Rescale,
    ToTensor
)

from Net.utils.common_utils import print_dict
from Net.utils.eval_utils import (LOSS,
                                  DET_LOSS,
                                  DES_LOSS,
                                  REP_SCORE,
                                  MATCH_SCORE,
                                  NN_MATCH_SCORE,
                                  NNT_MATCH_SCORE,
                                  NNR_MATCH_SCORE,
                                  SHOW,

                                  KP1,
                                  KP2,
                                  DESC1,
                                  DESC2,

                                  IMAGE_SIZE,
                                  TOP_K,
                                  HOMO_INV,
                                  KP_MT,
                                  DS_MT,
                                  DS_MR,

                                  l_loss,
                                  l_det_loss,
                                  l_des_loss,
                                  l_rep_score,
                                  l_match_score,
                                  l_collect_show,

                                  plot_keypoints)
from Net.utils.ignite_utils import AverageMetric, CollectMetric, AverageListMetric
from Net.utils.image_utils import select_keypoints
from Net.utils.model_utils import sample_descriptors


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
        transform += [Rescale((320, 640))]
        include_originals = True

    transform += [ToTensor()]

    return HPatchesDataset(root_path=cfg.DATASET.view.root,
                           csv_file=csv_file,
                           transform=transforms.Compose(transform),
                           include_originals=include_originals)


def attach_metrics(engine):
    # AverageMetric(l_loss, cfg.TRAIN.LOG_INTERVAL).attach(engine, LOSS)
    AverageMetric(l_det_loss, cfg.TRAIN.LOG_INTERVAL).attach(engine, DET_LOSS)
    # AverageMetric(l_des_loss, cfg.TRAIN.LOG_INTERVAL).attach(engine, DES_LOSS)
    AverageMetric(l_rep_score, cfg.TRAIN.LOG_INTERVAL).attach(engine, REP_SCORE)
    # AverageListMetric(l_match_score, cfg.TRAIN.LOG_INTERVAL).attach(engine, MATCH_SCORE)


def output_metrics(writer, data_engine, state_engine, tag):
    """
    :param writer: SummaryWriter
    :param data_engine: Engine to take data from
    :param state_engine: Engine to take current state from
    :param tag: Category to write data to
    """
    # writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{DET_LOSS}", data_engine.state.metrics[DET_LOSS], state_engine.state.iteration)
    # writer.add_scalar(f"{tag}/{DES_LOSS}", data_engine.state.metrics[DES_LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{REP_SCORE}", data_engine.state.metrics[REP_SCORE], state_engine.state.iteration)

    t_ms, nn_ms, nnt_ms, nnr_ms = data_engine.state.metrics[MATCH_SCORE]
    writer.add_scalar(f"{tag}/{MATCH_SCORE}", t_ms, state_engine.state.iteration)
    # writer.add_scalar(f"{tag}/{NN_MATCH_SCORE}", nn_ms, state_engine.state.iteration)
    # writer.add_scalar(f"{tag}/{NNT_MATCH_SCORE}", nnt_ms, state_engine.state.iteration)
    # writer.add_scalar(f"{tag}/{NNR_MATCH_SCORE}", nnr_ms, state_engine.state.iteration)


def prepare_output_dict(batch, endpoint, device):
    return {
        LOSS: endpoint[LOSS],
        DET_LOSS: endpoint[DET_LOSS],
        # DES_LOSS: endpoint[DES_LOSS],

        KP1: endpoint[KP1],
        KP2: endpoint[KP2],

        # DESC1: endpoint[DESC1],
        # DESC2: endpoint[DESC2],

        IMAGE_SIZE: batch[IMAGE1].size(),
        TOP_K: cfg.LOSS.TOP_K,
        KP_MT: cfg.METRIC.DET_THRESH,
        DS_MT: cfg.METRIC.DES_THRESH,
        DS_MR: cfg.METRIC.DES_RATIO,

        HOMO: batch[HOMO].to(device),
        HOMO_INV: endpoint[HOMO_INV],
    }


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

    model = Net().to(device)
    # model = Net(cfg.MODEL.GRID_SIZE, cfg.MODEL.DESCRIPTOR_SIZE).to(device)

    # det_criterion = HomoRFMSELoss(cfg.LOSS.NMS_THRESH, cfg.LOSS.NMS_K_SIZE,
    #                             cfg.LOSS.TOP_K,
    #                             cfg.LOSS.GAUSS_K_SIZE, cfg.LOSS.GAUSS_SIGMA)
    det_criterion = HomoMSELoss(cfg.LOSS.NMS_THRESH, cfg.LOSS.NMS_K_SIZE,
                                  cfg.LOSS.TOP_K,
                                  cfg.LOSS.GAUSS_K_SIZE, cfg.LOSS.GAUSS_SIGMA)
    # des_criterion = HomoHingeLoss(cfg.MODEL.GRID_SIZE, cfg.LOSS.POS_LAMBDA,
    #                               cfg.LOSS.POS_MARGIN, cfg.LOSS.NEG_MARGIN)

    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    def iteration(engine, batch):
        image1, image2, homo = (
            batch[IMAGE1].to(device),
            batch[IMAGE2].to(device),
            batch[HOMO].to(device)
        )
        homo_inv = homo.inverse()

        score1 = model(image1)
        score2 = model(image2)

        det_loss1, kp1, vis_mask1 = det_criterion(score1, score2, homo)
        det_loss2, kp2, vis_mask2 = det_criterion(score2, score1, homo_inv)

        det_loss = (det_loss1 + det_loss2) / 2
        loss = det_loss

        return {
            LOSS: loss,
            DET_LOSS: det_loss,

            HOMO_INV: homo_inv,

            KP1: kp1,
            KP2: kp2,
        }

    def train_iteration(engine, batch):
        model.train()

        optimizer.zero_grad()

        endpoint = iteration(engine, batch)
        endpoint[LOSS].backward()

        optimizer.step()

        return prepare_output_dict(batch, endpoint, device)

    def validation_iteration(engine, batch):
        model.eval()

        endpoint = iteration(engine, batch)

        return prepare_output_dict(batch, endpoint, device)

    def validation_show_iteration(engine, batch):
        model.eval()

        endpoint = iteration(engine, batch)

        return {
            S_IMAGE1: batch[S_IMAGE1],
            S_IMAGE2: batch[S_IMAGE2],

            KP1: endpoint[KP1],
            KP2: endpoint[KP2],
        }

    trainer = Engine(train_iteration)
    validator = Engine(validation_iteration)
    validator_show = Engine(validation_show_iteration)

    # TODO. Save the best model by providing score function and include it in files name. LATER
    checkpoint_saver = ModelCheckpoint(checkpoint_dir, "my", save_interval=1, n_saved=3)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, {'model': model})

    """
    Visualisation utils, logging and metrics
    """

    writer = SummaryWriter(logdir=log_dir)

    attach_metrics(trainer)
    attach_metrics(validator)
    CollectMetric(l_collect_show).attach(validator_show, SHOW)

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
            plot_keypoints(writer, engine.state.epoch, validator_show.state.metrics[SHOW])

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
