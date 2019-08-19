import os
from easydict import EasyDict

import torch
from Net.source.nn.model import NetVGG, NetVGGDebug
from Net.source.nn.criterion import MSELoss, HardQuadTripletSOSRLoss
from torch.optim import Adam

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.contrib.handlers.custom_events import CustomPeriodicEvent
from ignite.handlers import ModelCheckpoint

from Net.experiments.base_experiment import BaseExperiment
from Net.source.hpatches_dataset import (
    HPatchesDataset,
    Grayscale,
    Normalize,
    RandomCrop,
    Rescale,
    ToTensor,

    IMAGE1,
    IMAGE2,
    HOMO12,
    HOMO21,
    S_IMAGE1,
    S_IMAGE2
)
from Net.source.utils.image_utils import select_keypoints, warp_points
from Net.source.utils.model_utils import sample_descriptors
from Net.source.utils.ignite_utils import PeriodicMetric, AveragePeriodicMetric, AveragePeriodicListMetric, \
    CollectMetric
from Net.source.utils.eval_utils import repeatability_score, match_score, plot_keypoints_and_descriptors

"""
Dataset and loader keys
"""
TRAIN_LOADER = 'train_loader'
VAL_LOADER = 'val_loader'
SHOW_LOADER = 'show_loader'

"""
Model keys
"""
MODEL = 'model'

"""
Criterion keys
"""
DET_CRITERION = 'det_criterion'
DES_CRITERION = 'des_criterion'

"""
Optimizer keys
"""
OPTIMIZER = 'optimizer'

"""
Engine keys
"""
TRAIN_ENGINE = 'train_engine'
VAL_ENGINE = 'val_engine'
SHOW_ENGINE = 'show_engine'

"""
Endpoint keys
"""
LOSS = 'loss'
DET_LOSS = 'det_loss'
DES_LOSS = 'des_loss'

KP1 = 'kp1'
KP2 = 'kp2'
W_KP1 = 'w_kp1'
W_KP2 = 'w_kp2'
KP1_DESC = 'kp1_desc'
KP2_DESC = 'kp2_desc'
DESC1 = 'desc1'
DESC2 = 'desc2'

DEBUG1 = 'debug1'
DEBUG2 = 'debug2'
SCORE1 = 'score1'
SCORE2 = 'score2'

"""
Metric keys
"""
REP_SCORE = 'repeatability_score'
MATCH_SCORE = 'match_score'
NN_MATCH_SCORE = 'nearest_neighbour_match_score'
NNT_MATCH_SCORE = 'nearest_neighbour_thresh_match_score'
NNR_MATCH_SCORE = 'nearest_neighbour_ratio_match_score'
SHOW = 'show'

CHECKPOINT_PREFIX = "my"


# noinspection PyMethodMayBeStatic
class TrainExperiment(BaseExperiment):
    """
    Current main train experiment
    """

    def __init__(self, device, log_dir=None, checkpoint_dir=None, checkpoint_iter=None):
        super().__init__(device, log_dir, checkpoint_dir, checkpoint_iter)

        self.writer = None

    def get_dataset_settings(self):
        ds = EasyDict()

        ds.DATASET_ROOT = "../data/hpatch_v_sequence"
        ds.DATASET_MEAN = 0.4230204841414801
        ds.DATASET_STD = 0.25000138349993173
        ds.TRAIN_CSV = "train.csv"
        ds.VAL_CSV = "val.csv"
        ds.SHOW_CSV = "show.csv"
        ds.ANALYZE_CSV = "analyze.csv"

        return ds

    def get_loaders_settings(self):
        ls = EasyDict()

        ls.TRAIN_BATCH_SIZE = 16
        ls.VAL_BATCH_SIZE = 16
        ls.SHOW_BATCH_SIZE = 1
        ls.ANALYZE_BATCH_SIZE = 1
        ls.NUM_WORKERS = 8

        return ls

    def get_model_settings(self):
        ms = EasyDict()

        ms.GRID_SIZE = 8
        ms.DESCRIPTOR_SIZE = 32

        return ms

    def get_criterion_settings(self):
        cs = EasyDict()

        cs.DES_LAMBDA = 1
        cs.MARGIN = 1
        cs.NUM_NEG = 4
        cs.SOS_NEG = 8

        cs.DET_LAMBDA = 50

        cs.NMS_THRESH = 0.0
        cs.NMS_K_SIZE = 5

        cs.TOP_K = 512

        cs.GAUSS_K_SIZE = 15
        cs.GAUSS_SIGMA = 0.5

        return cs

    def get_log_settings(self):
        ls = EasyDict()

        ls.TRAIN = EasyDict()
        ls.TRAIN.LOSS_LOG_INTERVAL = 2
        ls.TRAIN.METRIC_LOG_INTERVAL = 8

        ls.VAL = EasyDict()
        ls.VAL.LOG_INTERVAL = 8

        ls.SHOW = EasyDict()
        ls.SHOW.LOG_INTERVAL = 5

        return ls

    def get_experiment_settings(self):
        es = EasyDict()

        es.NUM_EPOCHS = 3000

        return es

    def get_metric_settings(self):
        ms = EasyDict()

        ms.DET_THRESH = 5.0
        ms.DES_THRESH = 1.0
        ms.DES_RATIO = 0.7

        return ms

    def get_checkpoint_settings(self):
        cs = EasyDict()

        cs.SAVE_INTERVAL = 1000
        cs.N_SAVED = 3

        return cs

    @staticmethod
    def get_dataset(root_path, csv_file, transform, include_sources):
        return HPatchesDataset(root_path=root_path,
                               csv_file=csv_file,
                               transform=transforms.Compose(transform),
                               include_sources=include_sources)

    def init_loaders(self):
        ds = self.get_dataset_settings()
        ls = self.get_loaders_settings()

        train_transform = [Grayscale(),
                           Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                           Rescale((960, 1280)),
                           RandomCrop((720, 960)),
                           Rescale((240, 320)),
                           ToTensor()]

        val_transform = [Grayscale(),
                         Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                         Rescale((960, 1280)),
                         Rescale((240, 320)),
                         ToTensor()]

        show_transform = [Grayscale(),
                          Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                          Rescale((960, 1280)),
                          Rescale((320, 640)),
                          ToTensor()]

        self.loaders[TRAIN_LOADER] = DataLoader(
            TrainExperiment.get_dataset(ds.DATASET_ROOT, ds.TRAIN_CSV, train_transform, False),
            batch_size=ls.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=ls.NUM_WORKERS)
        self.loaders[VAL_LOADER] = DataLoader(
            TrainExperiment.get_dataset(ds.DATASET_ROOT, ds.VAL_CSV, val_transform, False),
            batch_size=ls.VAL_BATCH_SIZE,
            shuffle=True,
            num_workers=ls.NUM_WORKERS)
        self.loaders[SHOW_LOADER] = DataLoader(
            TrainExperiment.get_dataset(ds.DATASET_ROOT, ds.SHOW_CSV, show_transform, True),
            batch_size=ls.SHOW_BATCH_SIZE)

    def init_models(self):
        ms = self.get_model_settings()
        self.models[MODEL] = NetVGG(ms.GRID_SIZE, ms.DESCRIPTOR_SIZE).to(self.device)

    def init_criterions(self):
        ms = self.get_model_settings()
        cs = self.get_criterion_settings()

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = HardQuadTripletSOSRLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG, cs.SOS_NEG,
                                                                 cs.DES_LAMBDA)

    def init_optimizers(self):
        self.optimizers[OPTIMIZER] = Adam(self.models[MODEL].parameters())

    def load_checkpoint(self):
        if self.checkpoint_iter is not None:
            if MODEL in self.models:
                model_path = os.path.join(self.checkpoint_dir,
                                          f"{CHECKPOINT_PREFIX}_{MODEL}_{self.checkpoint_iter}.pth")
                self.models[MODEL].load_state_dict(torch.load(model_path, map_location=self.device))

            if OPTIMIZER in self.optimizers:
                optimizer_path = os.path.join(self.checkpoint_dir,
                                              f"{CHECKPOINT_PREFIX}_{OPTIMIZER}_{self.checkpoint_iter}.pth")
                self.optimizers[OPTIMIZER].load_state_dict(torch.load(optimizer_path, map_location=self.device))

    def start_logging(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def iteration(self, engine, batch):
        model = self.models[MODEL]

        mse_criterion = self.criterions[DET_CRITERION]
        triplet_criterion = self.criterions[DES_CRITERION]

        ms = self.get_model_settings()

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, desc1 = model(image1)
        score2, desc2 = model(image2)

        det_loss1, kp1 = mse_criterion(score1, score2, homo12)
        det_loss2, kp2 = mse_criterion(score2, score1, homo21)

        kp1_desc = sample_descriptors(desc1, kp1, ms.GRID_SIZE)
        kp2_desc = sample_descriptors(desc2, kp2, ms.GRID_SIZE)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        des_loss1 = triplet_criterion(kp1, w_kp1, kp1_desc, desc2, homo12)
        des_loss2 = triplet_criterion(kp2, w_kp2, kp2_desc, desc1, homo21)

        det_loss = (det_loss1 + det_loss2) / 2
        des_loss = (des_loss1 + des_loss2) / 2

        loss = det_loss + des_loss

        return {
            LOSS: loss,
            DET_LOSS: det_loss,
            DES_LOSS: des_loss,

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc
        }

    def inference(self, engine, batch):
        model = self.models[MODEL]

        ms = self.get_model_settings()
        ls = self.get_criterion_settings()

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, desc1 = model(image1)
        score2, desc2 = model(image2)

        _, kp1 = select_keypoints(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
        _, kp2 = select_keypoints(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

        kp1_desc = sample_descriptors(desc1, kp1, ms.GRID_SIZE)
        kp2_desc = sample_descriptors(desc2, kp2, ms.GRID_SIZE)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        return {
            S_IMAGE1: batch[S_IMAGE1],
            S_IMAGE2: batch[S_IMAGE2],

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc,
        }

    def init_engines(self):
        model = self.models[MODEL]

        optimizer = self.optimizers[OPTIMIZER]

        def train_iteration(engine, batch):
            model.train()

            with torch.autograd.set_detect_anomaly(True):
                endpoint = self.iteration(engine, batch)

                optimizer.zero_grad()
                endpoint[LOSS].backward()
                optimizer.step()

            return endpoint

        def val_iteration(engine, batch):
            model.eval()

            with torch.no_grad():
                endpoint = self.iteration(engine, batch)

            return endpoint

        def show_iteration(engine, batch):
            model.eval()

            with torch.no_grad():
                endpoint = self.inference(engine, batch)

            return endpoint

        self.engines[TRAIN_ENGINE] = Engine(train_iteration)
        self.engines[VAL_ENGINE] = Engine(val_iteration)
        self.engines[SHOW_ENGINE] = Engine(show_iteration)

    def bind_events(self):
        train_engine = self.engines[TRAIN_ENGINE]
        val_engine = self.engines[VAL_ENGINE]
        show_engine = self.engines[SHOW_ENGINE]

        if self.checkpoint_dir is not None:
            check_s = self.get_checkpoint_settings()

            checkpoint_saver = ModelCheckpoint(self.checkpoint_dir, CHECKPOINT_PREFIX,
                                               save_interval=check_s.SAVE_INTERVAL, n_saved=check_s.N_SAVED)

            model = self.models[MODEL]
            optimizers = self.optimizers[OPTIMIZER]

            train_engine.add_event_handler(Events.ITERATION_COMPLETED, checkpoint_saver, {MODEL: model,
                                                                                          OPTIMIZER: optimizers})

        ls = self.get_log_settings()
        cs = self.get_criterion_settings()
        ms = self.get_metric_settings()

        def l_loss(x):
            return x[LOSS]

        def l_det_loss(x):
            return x[DET_LOSS]

        def l_des_loss(x):
            return x[DES_LOSS]

        def l_rep_score(x):
            return repeatability_score(x[KP1], x[W_KP2], x[KP2], cs.TOP_K, ms.DET_THRESH)[0]

        def l_match_score(x):
            return match_score(x[KP1], x[W_KP2], x[KP2], x[KP1_DESC], x[KP2_DESC],
                               cs.TOP_K, ms.DET_THRESH, ms.DES_THRESH, ms.DES_RATIO)

        def l_collect_show(x):
            return x[S_IMAGE1][0], x[S_IMAGE2][0], x[KP1][0], x[W_KP2][0], x[KP2][0], x[KP1_DESC][0], x[KP2_DESC][0]

        # Train metrics
        AveragePeriodicMetric(l_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, LOSS)
        AveragePeriodicMetric(l_det_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, DET_LOSS)
        AveragePeriodicMetric(l_des_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, DES_LOSS)
        PeriodicMetric(l_rep_score, ls.TRAIN.METRIC_LOG_INTERVAL).attach(train_engine, REP_SCORE)
        PeriodicMetric(l_match_score, ls.TRAIN.METRIC_LOG_INTERVAL).attach(train_engine, MATCH_SCORE)

        # Val metrics
        AveragePeriodicMetric(l_loss).attach(val_engine, LOSS)
        AveragePeriodicMetric(l_det_loss).attach(val_engine, DET_LOSS)
        AveragePeriodicMetric(l_des_loss).attach(val_engine, DES_LOSS)
        AveragePeriodicMetric(l_rep_score).attach(val_engine, REP_SCORE)
        AveragePeriodicListMetric(l_match_score).attach(val_engine, MATCH_SCORE)

        # Show metrics
        CollectMetric(l_collect_show).attach(show_engine, SHOW)

        tle = CustomPeriodicEvent(n_iterations=ls.TRAIN.LOSS_LOG_INTERVAL)
        tme = CustomPeriodicEvent(n_iterations=ls.TRAIN.METRIC_LOG_INTERVAL)
        ve = CustomPeriodicEvent(n_iterations=ls.VAL.LOG_INTERVAL)
        se = CustomPeriodicEvent(n_epochs=ls.SHOW.LOG_INTERVAL)

        tle.attach(train_engine)
        tme.attach(train_engine)
        ve.attach(train_engine)
        se.attach(train_engine)

        val_loader = self.loaders[VAL_LOADER]
        show_loader = self.loaders[SHOW_LOADER]

        def output_losses(writer, data_engine, state_engine, tag):
            writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)
            writer.add_scalar(f"{tag}/{DET_LOSS}", data_engine.state.metrics[DET_LOSS], state_engine.state.iteration)
            writer.add_scalar(f"{tag}/{DES_LOSS}", data_engine.state.metrics[DES_LOSS], state_engine.state.iteration)

        def output_metrics(writer, data_engine, state_engine, tag):
            writer.add_scalar(f"{tag}/{REP_SCORE}", data_engine.state.metrics[REP_SCORE], state_engine.state.iteration)

            t_ms, nn_ms, nnt_ms, nnr_ms = data_engine.state.metrics[MATCH_SCORE]
            writer.add_scalar(f"{tag}/{MATCH_SCORE}", t_ms, state_engine.state.iteration)
            writer.add_scalar(f"{tag}/{NN_MATCH_SCORE}", nn_ms, state_engine.state.iteration)
            writer.add_scalar(f"{tag}/{NNT_MATCH_SCORE}", nnt_ms, state_engine.state.iteration)
            writer.add_scalar(f"{tag}/{NNR_MATCH_SCORE}", nnr_ms, state_engine.state.iteration)

        @train_engine.on(tle._periodic_event_completed)
        def on_tle(engine):
            output_losses(self.writer, train_engine, train_engine, "train")

        @train_engine.on(tme._periodic_event_completed)
        def on_tme(engine):
            output_metrics(self.writer, train_engine, train_engine, "train")

        @train_engine.on(ve._periodic_event_completed)
        def on_ve(engine):
            val_engine.run(val_loader)

            output_losses(self.writer, val_engine, train_engine, "val")
            output_metrics(self.writer, val_engine, train_engine, "val")

        @train_engine.on(se._periodic_event_completed)
        def on_se(engine):
            show_engine.run(show_loader)

            plot_keypoints_and_descriptors(self.writer, train_engine.state.epoch, show_engine.state.metrics[SHOW],
                                           cs.TOP_K, ms.DET_THRESH)

    def run_experiment(self):
        es = self.get_experiment_settings()

        train_engine = self.engines[TRAIN_ENGINE]
        train_loader = self.loaders[TRAIN_LOADER]

        train_engine.run(train_loader, max_epochs=es.NUM_EPOCHS)

    def stop_logging(self):
        self.writer.close()

    def analyze_inference(self):
        ds = self.get_dataset_settings()
        ls = self.get_loaders_settings()

        analyze_transform = [Grayscale(),
                             Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                             Rescale((960, 1280)),
                             Rescale((320, 640)),
                             ToTensor()]

        analyze_loader = DataLoader(
            TrainExperiment.get_dataset(os.path.join("../", ds.DATASET_ROOT),
                                        ds.ANALYZE_CSV, analyze_transform, True),
            batch_size=ls.ANALYZE_BATCH_SIZE)

        self.init_models()

        self.load_checkpoint()

        self.models[MODEL].eval()

        with torch.no_grad():
            batch = analyze_loader.__iter__().__next__()
            model = self.models[MODEL]

            ms = self.get_model_settings()
            ls = self.get_criterion_settings()

            image1, image2, homo12, homo21 = (
                batch[IMAGE1].to(self.device),
                batch[IMAGE2].to(self.device),
                batch[HOMO12].to(self.device),
                batch[HOMO21].to(self.device)
            )

            score1, desc1 = model(image1)
            score2, desc2 = model(image2)

            _, kp1 = select_keypoints(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
            _, kp2 = select_keypoints(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

            kp1_desc = sample_descriptors(desc1, kp1, ms.GRID_SIZE)
            kp2_desc = sample_descriptors(desc2, kp2, ms.GRID_SIZE)

            w_kp1 = warp_points(kp1, homo12)
            w_kp2 = warp_points(kp2, homo21)

            output = {
                S_IMAGE1: batch[S_IMAGE1],
                S_IMAGE2: batch[S_IMAGE2],

                HOMO12: homo12,
                HOMO21: homo21,

                SCORE1: score1,
                SCORE2: score2,

                DESC1: desc1,
                DESC2: desc2,

                KP1: kp1,
                KP2: kp2,

                W_KP1: w_kp1,
                W_KP2: w_kp2,

                KP1_DESC: kp1_desc,
                KP2_DESC: kp2_desc,
            }

        return output


class TrainExperimentDetector(TrainExperiment):

    def get_checkpoint_settings(self):
        cs = EasyDict()

        cs.SAVE_INTERVAL = 100
        cs.N_SAVED = 3

        return cs

    def init_models(self):
        ms = self.get_model_settings()
        self.models[MODEL] = NetVGGDebug(ms.GRID_SIZE, ms.DESCRIPTOR_SIZE).to(self.device)

    def iteration(self, engine, batch):
        model = self.models[MODEL]

        mse_criterion = self.criterions[DET_CRITERION]

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, _, _ = model(image1)
        score2, _, _ = model(image2)

        det_loss1, kp1 = mse_criterion(score1, score2, homo12)
        det_loss2, kp2 = mse_criterion(score2, score1, homo21)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        det_loss = (det_loss1 + det_loss2) / 2

        return {
            LOSS: det_loss,

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2
        }

    def inference(self, engine, batch):
        model = self.models[MODEL]

        ls = self.get_criterion_settings()

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, _, _ = model(image1)
        score2, _, _ = model(image2)

        _, kp1 = select_keypoints(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
        _, kp2 = select_keypoints(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        return {
            S_IMAGE1: batch[S_IMAGE1],
            S_IMAGE2: batch[S_IMAGE2],

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2
        }

    def bind_events(self):
        train_engine = self.engines[TRAIN_ENGINE]
        val_engine = self.engines[VAL_ENGINE]
        show_engine = self.engines[SHOW_ENGINE]

        if self.checkpoint_dir is not None:
            check_s = self.get_checkpoint_settings()

            checkpoint_saver = ModelCheckpoint(self.checkpoint_dir, CHECKPOINT_PREFIX,
                                               save_interval=check_s.SAVE_INTERVAL, n_saved=check_s.N_SAVED)

            model = self.models[MODEL]
            optimizers = self.optimizers[OPTIMIZER]

            train_engine.add_event_handler(Events.ITERATION_COMPLETED, checkpoint_saver, {MODEL: model,
                                                                                          OPTIMIZER: optimizers})

        ls = self.get_log_settings()
        cs = self.get_criterion_settings()
        ms = self.get_metric_settings()

        def l_loss(x):
            return x[LOSS]

        def l_rep_score(x):
            return repeatability_score(x[KP1], x[W_KP2], x[KP2], cs.TOP_K, ms.DET_THRESH)[0]

        def l_collect_show(x):
            return x[S_IMAGE1][0], x[S_IMAGE2][0], x[KP1][0], x[W_KP2][0], x[KP2][0], None, None

        # Train metrics
        AveragePeriodicMetric(l_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, LOSS)
        PeriodicMetric(l_rep_score, ls.TRAIN.METRIC_LOG_INTERVAL).attach(train_engine, REP_SCORE)

        # Val metrics
        AveragePeriodicMetric(l_loss).attach(val_engine, LOSS)
        AveragePeriodicMetric(l_rep_score).attach(val_engine, REP_SCORE)

        # Show metrics
        CollectMetric(l_collect_show).attach(show_engine, SHOW)

        tle = CustomPeriodicEvent(n_iterations=ls.TRAIN.LOSS_LOG_INTERVAL)
        tme = CustomPeriodicEvent(n_iterations=ls.TRAIN.METRIC_LOG_INTERVAL)
        ve = CustomPeriodicEvent(n_iterations=ls.VAL.LOG_INTERVAL)
        se = CustomPeriodicEvent(n_epochs=ls.SHOW.LOG_INTERVAL)

        tle.attach(train_engine)
        tme.attach(train_engine)
        ve.attach(train_engine)
        se.attach(train_engine)

        val_loader = self.loaders[VAL_LOADER]
        show_loader = self.loaders[SHOW_LOADER]

        def output_losses(writer, data_engine, state_engine, tag):
            writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)

        def output_metrics(writer, data_engine, state_engine, tag):
            writer.add_scalar(f"{tag}/{REP_SCORE}", data_engine.state.metrics[REP_SCORE], state_engine.state.iteration)

        @train_engine.on(tle._periodic_event_completed)
        def on_tle(engine):
            output_losses(self.writer, train_engine, train_engine, "train")

        @train_engine.on(tme._periodic_event_completed)
        def on_tme(engine):
            output_metrics(self.writer, train_engine, train_engine, "train")

        @train_engine.on(ve._periodic_event_completed)
        def on_ve(engine):
            val_engine.run(val_loader)

            output_losses(self.writer, val_engine, train_engine, "val")
            output_metrics(self.writer, val_engine, train_engine, "val")

        @train_engine.on(se._periodic_event_completed)
        def on_se(engine):
            show_engine.run(show_loader)

            plot_keypoints_and_descriptors(self.writer, train_engine.state.epoch, show_engine.state.metrics[SHOW],
                                           cs.TOP_K, ms.DET_THRESH)

    def analyze_inference(self):
        ds = self.get_dataset_settings()
        ls = self.get_loaders_settings()

        analyze_transform = [Grayscale(),
                             Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                             Rescale((960, 1280)),
                             Rescale((320, 640)),
                             ToTensor()]

        analyze_loader = DataLoader(
            TrainExperiment.get_dataset(os.path.join("../", ds.DATASET_ROOT),
                                        ds.ANALYZE_CSV, analyze_transform, True),
            batch_size=ls.ANALYZE_BATCH_SIZE)

        self.init_models()

        self.load_checkpoint()

        self.models[MODEL].eval()

        with torch.no_grad():
            batch = analyze_loader.__iter__().__next__()
            model = self.models[MODEL]

            ls = self.get_criterion_settings()

            image1, image2, homo12, homo21 = (
                batch[IMAGE1].to(self.device),
                batch[IMAGE2].to(self.device),
                batch[HOMO12].to(self.device),
                batch[HOMO21].to(self.device)
            )

            score1, _, debug1 = model(image1)
            score2, _, debug2 = model(image2)

            _, kp1 = select_keypoints(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
            _, kp2 = select_keypoints(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

            w_kp1 = warp_points(kp1, homo12)
            w_kp2 = warp_points(kp2, homo21)

            output = {
                S_IMAGE1: batch[S_IMAGE1],
                S_IMAGE2: batch[S_IMAGE2],

                DEBUG1: debug1,
                DEBUG2: debug2,

                HOMO12: homo12,
                HOMO21: homo21,

                SCORE1: score1,
                SCORE2: score2,

                KP1: kp1,
                KP2: kp2,

                W_KP1: w_kp1,
                W_KP2: w_kp2,
            }

        return output