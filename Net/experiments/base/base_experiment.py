import os
from abc import ABC, abstractmethod

from easydict import EasyDict


class BaseExperiment(ABC):

    def __init__(self, device, log_dir=None, checkpoint_dir=None, checkpoint_iter=None):
        self.device = device

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.checkpoint_iter = checkpoint_iter

        self.config = None

        self.loaders = {}
        self.models = {}
        self.criterions = {}
        self.optimizers = {}

        self.engines = {}

    @abstractmethod
    def init_config(self):
        ...

    @abstractmethod
    def init_loaders(self):
        ...

    @abstractmethod
    def init_models(self):
        ...

    @abstractmethod
    def init_criterions(self):
        ...

    @abstractmethod
    def init_optimizers(self):
        ...

    @abstractmethod
    def load_checkpoint(self):
        ...

    @abstractmethod
    def bind_checkpoint(self):
        ...

    @abstractmethod
    def start_logging(self):
        ...

    @abstractmethod
    def init_engines(self):
        ...

    @abstractmethod
    def bind_events(self):
        ...

    @abstractmethod
    def iteration(self, engine, batch):
        ...

    @abstractmethod
    def inference(self, engine, batch):
        ...

    @abstractmethod
    def run_experiment(self):
        ...

    @abstractmethod
    def stop_logging(self):
        ...

    def run(self):
        self.init_config()
        self.init_loaders()
        self.init_models()
        self.init_criterions()
        self.init_optimizers()

        self.load_checkpoint()

        self.start_logging()

        self.init_engines()
        self.bind_checkpoint()
        self.bind_events()

        self.run_experiment()

        self.stop_logging()

    @abstractmethod
    def analyze_inference(self):
        ...


class BaseConfig:

    def __init__(self):
        self.dataset = EasyDict()
        self.loader = EasyDict()
        self.model = EasyDict()
        self.criterion = EasyDict()
        self.log = EasyDict()
        self.experiment = EasyDict()
        self.metric = EasyDict()
        self.checkpoint = EasyDict()

        self.init_dataset()
        self.init_loader()
        self.init_model()
        self.init_criterion()
        self.init_log()
        self.init_experiment()
        self.init_metric()
        self.init_checkpoint()

    @abstractmethod
    def init_dataset(self):
        ...

    @abstractmethod
    def init_loader(self):
        ...

    @abstractmethod
    def init_model(self):
        ...

    @abstractmethod
    def init_criterion(self):
        ...

    @abstractmethod
    def init_log(self):
        ...

    @abstractmethod
    def init_experiment(self):
        ...

    @abstractmethod
    def init_metric(self):
        ...

    @abstractmethod
    def init_checkpoint(self):
        ...


class BaseMetricBinder(ABC):

    def __init__(self, config, writer):
        self.config = config
        self.writer = writer

    @abstractmethod
    def bind(self, engines, loaders):
        ...
