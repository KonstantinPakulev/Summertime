import os
from abc import ABC

import torch

from Net.experiments.base.base_experiment import BaseExperiment

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint


from Net.source.hpatches_dataset import (
    IMAGE1,
    IMAGE2,
    HOMO12,
    HOMO21,
    S_IMAGE1,
    S_IMAGE2
)


"""
Engine keys
"""
TRAIN_ENGINE = 'train_engine'
VAL_ENGINE = 'val_engine'
SHOW_ENGINE = 'show_engine'

"""
Dataset and loader keys
"""
TRAIN_LOADER = 'train_loader'
VAL_LOADER = 'val_loader'
SHOW_LOADER = 'show_loader'

"""
MAIN keys
"""
MODEL = 'model'
OPTIMIZER = 'optimizer'

CHECKPOINT_PREFIX = "my"

"""
Criterion keys
"""
DET_CRITERION = 'det_criterion'
DES_CRITERION = 'des_criterion'


"""
Batch keys
"""

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
WV_KP2_MASK = 'wv_kp2_mask'
WV_KP1_MASK = 'wv_kp1_mask'
KP1_DESC = 'kp1_desc'
KP2_DESC = 'kp2_desc'
DESC1 = 'desc1'
DESC2 = 'desc2'

DEBUG1 = 'debug1'
DEBUG2 = 'debug2'
SCORE1 = 'score1'
SCORE2 = 'score2'


class DefaultExperiment(BaseExperiment, ABC):

    def __init__(self, device, log_dir=None, checkpoint_dir=None, checkpoint_iter=None):
        super().__init__(device, log_dir, checkpoint_dir, checkpoint_iter)

        self.writer = None

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

    def bind_checkpoint(self):
        train_engine = self.engines[TRAIN_ENGINE]

        if self.checkpoint_dir is not None:
            check_s = self.config.checkpoint

            checkpoint_saver = ModelCheckpoint(self.checkpoint_dir, CHECKPOINT_PREFIX,
                                               save_interval=check_s.SAVE_INTERVAL, n_saved=check_s.N_SAVED)

            model = self.models[MODEL]
            optimizers = self.optimizers[OPTIMIZER]

            train_engine.add_event_handler(Events.ITERATION_COMPLETED, checkpoint_saver, {MODEL: model,
                                                                                          OPTIMIZER: optimizers})

    def run_experiment(self):
        es = self.config.experiment

        train_engine = self.engines[TRAIN_ENGINE]
        train_loader = self.loaders[TRAIN_LOADER]

        train_engine.run(train_loader, max_epochs=es.NUM_EPOCHS)

    def stop_logging(self):
        self.writer.close()