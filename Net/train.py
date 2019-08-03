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

from Net.nn.model import Net
from Net.nn.criterion import HomoMSELoss, HomoTripletLoss
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
                                  W_KP1,
                                  W_KP2,
                                  DESC1,
                                  DESC2,

                                  IMAGE_SIZE,
                                  TOP_K,
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
from Net.utils.image_utils import select_keypoints, warp_keypoints
from Net.utils.model_utils import sample_descriptors


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


def attach_metrics(engine, config):
    AverageMetric(l_loss, config.TRAIN.LOG_INTERVAL).attach(engine, LOSS)
    AverageMetric(l_det_loss, config.TRAIN.LOG_INTERVAL).attach(engine, DET_LOSS)
    AverageMetric(l_des_loss, config.TRAIN.LOG_INTERVAL).attach(engine, DES_LOSS)
    AverageMetric(l_rep_score, config.TRAIN.LOG_INTERVAL).attach(engine, REP_SCORE)
    AverageListMetric(l_match_score, config.TRAIN.LOG_INTERVAL).attach(engine, MATCH_SCORE)


def output_metrics(writer, data_engine, state_engine, tag):
    """
    :param writer: SummaryWriter
    :param data_engine: Engine to take data from
    :param state_engine: Engine to take current state from
    :param tag: Category to write data to
    """
    writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{DET_LOSS}", data_engine.state.metrics[DET_LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{DES_LOSS}", data_engine.state.metrics[DES_LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{REP_SCORE}", data_engine.state.metrics[REP_SCORE], state_engine.state.iteration)

    t_ms, nn_ms, nnt_ms, nnr_ms = data_engine.state.metrics[MATCH_SCORE]
    writer.add_scalar(f"{tag}/{MATCH_SCORE}", t_ms, state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{NN_MATCH_SCORE}", nn_ms, state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{NNT_MATCH_SCORE}", nnt_ms, state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{NNR_MATCH_SCORE}", nnr_ms, state_engine.state.iteration)


def prepare_output_dict(batch, endpoint, device, config):
    return {
        LOSS: endpoint[LOSS],
        DET_LOSS: endpoint[DET_LOSS],
        DES_LOSS: endpoint[DES_LOSS],

        KP1: endpoint[KP1],
        KP2: endpoint[KP2],

        W_KP1: endpoint[W_KP1],
        W_KP2: endpoint[W_KP2],

        DESC1: endpoint[DESC1],
        DESC2: endpoint[DESC2],

        IMAGE_SIZE: batch[IMAGE1].size(),

        TOP_K: config.LOSS.TOP_K,

        KP_MT: config.METRIC.DET_THRESH,
        DS_MT: config.METRIC.DES_THRESH,
        DS_MR: config.METRIC.DES_RATIO,
    }


def inference(model, batch, device, config):
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

    w_kp1 = warp_keypoints(kp1, homo12)
    w_kp2 = warp_keypoints(kp2, homo21)

    return {
        S_IMAGE1: batch[S_IMAGE1],
        S_IMAGE2: batch[S_IMAGE2],

        KP1: kp1,
        KP2: kp2,

        W_KP1: w_kp1,
        W_KP2: w_kp2,

        DESC1: kp1_desc,
        DESC2: kp2_desc,

        TOP_K: config.LOSS.TOP_K,

        KP_MT: config.METRIC.DET_THRESH,
    }


# TODO. Perform optimization and calculate all needed information after iteration procedure
def train(config, device, num_workers, log_dir, checkpoint_dir):
    """
    :param config: config to use
    :param device: cpu or gpu
    :param num_workers: number of workers to load the data
    :param log_dir: path to the directory to store tensorboard db
    :param checkpoint_dir: path to the directory to save checkpoints
    """
    """
    Dataset and data loaders preparation.
    """

    train_loader = DataLoader(h_patches_dataset(TRAIN, config),
                              batch_size=config.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(h_patches_dataset(VALIDATE, config),
                            batch_size=config.VAL.BATCH_SIZE,
                            shuffle=True,
                            num_workers=num_workers)

    val_show_loader = DataLoader(h_patches_dataset(VALIDATE_SHOW, config),
                                 batch_size=config.VAL_SHOW.BATCH_SIZE)

    """
    Model, optimizer and criterion settings. 
    Training and validation steps. 
    """

    model = Net(config.MODEL.DESCRIPTOR_SIZE).to(device)

    det_criterion = HomoMSELoss(config.LOSS.NMS_THRESH, config.LOSS.NMS_K_SIZE,
                                config.LOSS.TOP_K,
                                config.LOSS.GAUSS_K_SIZE, config.LOSS.GAUSS_SIGMA)
    des_criterion = HomoTripletLoss(config.MODEL.GRID_SIZE, config.LOSS.MARGIN, config.METRIC.DET_THRESH)

    optimizer = Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    def iteration(engine, batch):
        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(device),
            batch[IMAGE2].to(device),
            batch[HOMO12].to(device),
            batch[HOMO21].to(device)
        )

        score1, desc1 = model(image1)
        score2, desc2 = model(image2)

        det_loss1, kp1 = det_criterion(score1, score2, homo12)
        det_loss2, kp2 = det_criterion(score2, score1, homo21)

        des_loss1, kp1_desc, w_kp1 = des_criterion(kp1, desc1, desc2, homo12)
        des_loss2, kp2_desc, w_kp2 = des_criterion(kp2, desc2, desc1, homo21)

        det_loss = (det_loss1 + det_loss2) / 2 * config.LOSS.DET_LAMBDA
        des_loss = (des_loss1 + des_loss2) / 2 * config.LOSS.DES_LAMBDA

        loss = det_loss + des_loss

        return {
            LOSS: loss,
            DET_LOSS: det_loss,
            DES_LOSS: des_loss,

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            DESC1: kp1_desc,
            DESC2: kp2_desc
        }

    def train_iteration(engine, batch):
        model.train()

        optimizer.zero_grad()

        endpoint = iteration(engine, batch)
        endpoint[LOSS].backward()

        optimizer.step()

        return prepare_output_dict(batch, endpoint, device, config)

    def validation_iteration(engine, batch):
        model.eval()

        with torch.no_grad():
            endpoint = iteration(engine, batch)

        return prepare_output_dict(batch, endpoint, device, config)

    def validation_show_iteration(engine, batch):
        model.eval()

        with torch.no_grad():
            endpoint = inference(model, batch, device, config)

        return endpoint

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

    attach_metrics(trainer, config)
    attach_metrics(validator, config)
    CollectMetric(l_collect_show).attach(validator_show, SHOW)

    """
    Registering callbacks for sending summary to tensorboard
    """
    epoch_timer = Timer(average=False)
    batch_timer = Timer(average=True)

    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iteration_completed(engine):
        if engine.state.iteration % config.TRAIN.LOG_INTERVAL == 0:
            output_metrics(writer, engine, engine, "train")

        if engine.state.iteration % config.VAL.LOG_INTERVAL == 0:
            validator.run(val_loader)
            output_metrics(writer, validator, engine, "val")

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        if engine.state.epoch % config.VAL_SHOW.LOG_INTERVAL == 0:
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

    trainer.run(train_loader, max_epochs=config.TRAIN.NUM_EPOCHS)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    from Net.train_config import cfg as train_config
    from Net.test_config import cfg as test_config

    if args.config == 'test':
        _config = test_config
    else:
        _config = train_config

    print_dict(_config)

    _device, _num_workers = (torch.device('cuda'), 8) if torch.cuda.is_available() else (torch.device('cpu'), 0)
    _log_dir = args.log_dir
    _checkpoint_dir = args.checkpoint_dir

    train(_config, _device, _num_workers, _log_dir, _checkpoint_dir)
