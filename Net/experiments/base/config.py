from easydict import EasyDict

from Net.experiments.base.base_experiment import BaseConfig


class MainConfig(BaseConfig):

    def init_dataset(self):
        self.dataset.DATASET_ROOT = "../data/hpatch_v_sequence"
        self.dataset.DATASET_MEAN = 0.4230204841414801
        self.dataset.DATASET_STD = 0.25000138349993173
        self.dataset.TRAIN_CSV = "train.csv"
        self.dataset.VAL_CSV = "val.csv"
        self.dataset.SHOW_CSV = "show.csv"
        self.dataset.ANALYZE_CSV = "analyze.csv"

    def init_loader(self):
        self.loader.TRAIN_BATCH_SIZE = 16
        self.loader.VAL_BATCH_SIZE = 16
        self.loader.SHOW_BATCH_SIZE = 1
        self.loader.ANALYZE_BATCH_SIZE = 1
        self.loader.NUM_WORKERS = 8

    def init_model(self):
        self.model.GRID_SIZE = 8
        self.model.DESCRIPTOR_SIZE = 32
        self.model.NMS_KERNEL_SIZE = 15

    def init_criterion(self):
        self.criterion.DES_LAMBDA = 1
        self.criterion.MARGIN = 1
        self.criterion.NUM_NEG = 1
        self.criterion.SOS_NEG = 4

        self.criterion.DET_LAMBDA = 120

        self.criterion.NMS_THRESH = 0.0
        self.criterion.NMS_K_SIZE = 5

        self.criterion.TOP_K = 512

        self.criterion.GAUSS_K_SIZE = 15
        self.criterion.GAUSS_SIGMA = 0.5

    def init_log(self):
        self.log.TRAIN = EasyDict()
        self.log.TRAIN.LOSS_LOG_INTERVAL = 2
        self.log.TRAIN.METRIC_LOG_INTERVAL = 8

        self.log.VAL = EasyDict()
        self.log.VAL.LOG_INTERVAL = 20

        self.log.SHOW = EasyDict()
        self.log.SHOW.LOG_INTERVAL = 5

    def init_experiment(self):
        self.experiment.NUM_EPOCHS = 3000

    def init_metric(self):
        self.metric.DET_THRESH = 5.0
        self.metric.DES_THRESH = 1.0
        self.metric.DES_RATIO = 0.7

    def init_checkpoint(self):
        self.checkpoint.SAVE_INTERVAL = 1000
        self.checkpoint.N_SAVED = 3


class DetectorConfig(MainConfig):

    def init_checkpoint(self):
        self.checkpoint.SAVE_INTERVAL = 100
        self.checkpoint.N_SAVED = 3


class DebugConfig(MainConfig):

    def init_dataset(self):
        self.dataset.DATASET_ROOT = "../data/hpatch_v_sequence"
        self.dataset.DATASET_MEAN = 0.4230204841414801
        self.dataset.DATASET_STD = 0.25000138349993173
        self.dataset.TRAIN_CSV = "debug.csv"
        self.dataset.VAL_CSV = "debug.csv"
        self.dataset.SHOW_CSV = "debug.csv"
        self.dataset.ANALYZE_CSV = "analyze.csv"

    def init_loader(self):
        self.loader.TRAIN_BATCH_SIZE = 1
        self.loader.VAL_BATCH_SIZE = 1
        self.loader.SHOW_BATCH_SIZE = 1
        self.loader.ANALYZE_BATCH_SIZE = 1
        self.loader.NUM_WORKERS = 0

    def init_model(self):
        self.model.GRID_SIZE = 8
        self.model.DESCRIPTOR_SIZE = 4
        self.model.NMS_KERNEL_SIZE = 15

    def init_log(self):
        self.log.TRAIN = EasyDict()
        self.log.TRAIN.LOSS_LOG_INTERVAL = 1
        self.log.TRAIN.METRIC_LOG_INTERVAL = 1

        self.log.VAL = EasyDict()
        self.log.VAL.LOG_INTERVAL = 1

        self.log.SHOW = EasyDict()
        self.log.SHOW.LOG_INTERVAL = 1

    def init_experiment(self):
        self.experiment.NUM_EPOCHS = 100

    def init_checkpoint(self):
        self.checkpoint.SAVE_INTERVAL = 19
        self.checkpoint.N_SAVED = 5
