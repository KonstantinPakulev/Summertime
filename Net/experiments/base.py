from abc import ABC, abstractmethod

from easydict import EasyDict

import torch
from ignite.engine import Engine


class Experiment(ABC):

    def __init__(self, device, log_dir=None, checkpoint_dir=None, checkpoint_iter=None):
        self.device = device

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_iter = checkpoint_iter

        self.models_configs = self.get_models_configs()
        self.criterions_configs = self.get_criterions_configs()
        self.experiment_config = self.get_experiments_configs()

        self.models = self.get_models(self.models_configs)
        self.criterions = self.get_criterions(self.models_configs, self.criterions_configs)
        self.optimizers = self.get_optimizers(self.models)

        self.loop = self.get_loop()

    @abstractmethod
    def get_models_configs(self):
        ...

    @abstractmethod
    def get_criterions_configs(self):
        ...

    @abstractmethod
    def get_experiments_configs(self):
        ...

    @abstractmethod
    def get_models(self, models_configs):
        ...

    @abstractmethod
    def get_criterions(self, models_configs, criterions_configs):
        ...

    @abstractmethod
    def get_optimizers(self, models):
        ...

    @abstractmethod
    def load_checkpoints(self, models, optimizers):
        ...

    @abstractmethod
    def get_loop(self):
        ...

    @abstractmethod
    def bind_checkpoints(self, engine, models, optimizers):
        ...

    @abstractmethod
    def start_logging(self):
        ...

    @abstractmethod
    def end_logging(self):
        ...

    def run(self):
        self.load_checkpoints(self.models, self.optimizers)
        self.bind_checkpoints(self.loop.engine, self.models, self.optimizers)

        self.start_logging()

        self.loop.run()

        self.end_logging()


class Loop(ABC):

    def __init__(self, device, dataset_config, loader_config, models, criterion, optimizer,
                 models_configs, criterions_configs, log_config, metric_config, experiment_config,
                 train_mode):
        self.device = device

        self.dataset = self.get_dataset(dataset_config)
        self.loader = self.get_loader(self.dataset, loader_config)

        self.models = models
        self.criterions = criterion
        self.optimizers = optimizer

        self.models_configs = models_configs
        self.criterions_configs = criterions_configs
        self.log_config = log_config
        self.metric_config = metric_config
        self.experiment_config = experiment_config

        self.train_mode = train_mode

        def iteration(engine, batch):
            if self.train_mode:
                self.models.train()
                endpoint = self.forward(engine, batch)
            else:
                self.models.eval()
                with torch.no_grad():
                    endpoint = self.forward(engine, batch)

            self.calculate_losses(engine, endpoint)
            self.finalize_endpoint(engine, batch, endpoint)
            self.step(engine, endpoint)

            return endpoint

        self.engine = Engine(iteration)
        self.bind_metrics(self.engine)

    @abstractmethod
    def get_dataset(self, dataset_config):
        ...

    @abstractmethod
    def get_loader(self, dataset, loader_config):
        ...

    @abstractmethod
    def forward(self, engine, batch):
        ...

    @abstractmethod
    def calculate_losses(self, engine, endpoint):
        ...

    @abstractmethod
    def finalize_endpoint(self, engine, batch, endpoint):
        ...

    @abstractmethod
    def step(self, engine, endpoint):
        ...

    @abstractmethod
    def bind_metrics(self, engine):
        ...

    def run(self):
        self.engine.run(self.loader, max_epochs=self.experiment_config.NUM_EPOCHS)


class Config:

    def __init__(self):
        self.config = EasyDict()

    @abstractmethod
    def get(self):
        ...
