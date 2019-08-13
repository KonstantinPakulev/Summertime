from abc import ABC, abstractmethod

import os
import shutil


class BaseExperiment(ABC):

    def __init__(self, device, log_dir=None, checkpoint_dir=None, checkpoint_iter=None):
        self.device = device

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.checkpoint_iter = checkpoint_iter

        self.loaders = {}
        self.models = {}
        self.criterions = {}
        self.optimizers = {}

        self.engines = {}

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
    def start_logging(self):
        ...

    @abstractmethod
    def init_engines(self):
        ...

    @abstractmethod
    def bind_events(self):
        ...

    @abstractmethod
    def run_experiment(self):
        ...

    @abstractmethod
    def stop_logging(self):
        ...

    def run(self):
        self.init_loaders()
        self.init_models()
        self.init_criterions()
        self.init_optimizers()

        self.load_checkpoint()

        self.start_logging()

        self.init_engines()
        self.bind_events()

        self.run_experiment()

        self.stop_logging()

    @abstractmethod
    def analyze_inference(self):
        ...
