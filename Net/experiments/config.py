from Net.experiments.base import Config

"""
Dataset configs
"""


class TrainDatasetConfig(Config):

    def __init__(self):
        super().__init__()
        self.config.DATASET_ROOT = "../data/hpatch_v_sequence"

    def get(self):
        self.config.CSV = "train.csv"
        self.config.BRIGHTNESS_CHANGE = 0.1
        self.config.CONTRAST_CHANGE = 0.1
        self.config.HEIGHT = 240
        self.config.WIDTH = 320
        self.config.INCLUDE_SOURCES = False
        return self.config


class ValDatasetConfig(TrainDatasetConfig):

    def get(self):
        self.config.CSV = "val.csv"
        self.config.HEIGHT = 240
        self.config.WIDTH = 320
        self.config.INCLUDE_SOURCES = False
        return self.config


class ShowDatasetConfig(TrainDatasetConfig):

    def get(self):
        self.config.CSV = "show.csv"
        self.config.HEIGHT = 480
        self.config.WIDTH = 640
        self.config.INCLUDE_SOURCES = True
        return self.config


class TestDatasetConfig(TrainDatasetConfig):

    def get(self):
        self.config.CSV = "test.csv"
        self.config.HEIGHT = 240
        self.config.WIDTH = 320
        self.config.INCLUDE_SOURCES = True
        return self.config


class AnalyzeDatasetConfig(Config):

    def __init__(self):
        super().__init__()
        self.config.DATASET_ROOT = "../../data/hpatch_v_sequence"

    def get(self):
        self.config.CSV = "analyze.csv"
        self.config.HEIGHT = 480
        self.config.WIDTH = 640
        self.config.INCLUDE_SOURCES = True
        return self.config


class DebugTrainValDatasetConfig(TrainDatasetConfig):

    def get(self):
        self.config.CSV = "debug.csv"
        self.config.HEIGHT = 240
        self.config.WIDTH = 320
        self.config.INCLUDE_SOURCES = False
        return self.config


class DebugShowDatasetConfig(TrainDatasetConfig):

    def get(self):
        self.config.CSV = "debug.csv"
        self.config.HEIGHT = 480
        self.config.WIDTH = 640
        self.config.INCLUDE_SOURCES = True
        return self.config


"""
Loader configs
"""


class TrainValLoaderConfig(Config):

    def __init__(self):
        super().__init__()
        self.config.BATCH_SIZE = 16
        self.config.SHUFFLE = True
        self.config.NUM_WORKERS = 8

    def get(self):
        return self.config


class ShowLoaderConfig(Config):

    def __init__(self):
        super().__init__()
        self.config.BATCH_SIZE = 1
        self.config.SHUFFLE = False
        self.config.NUM_WORKERS = 0

    def get(self):
        return self.config


class TestLoaderConfig(Config):

    def __init__(self):
        super().__init__()
        self.config.BATCH_SIZE = 1
        self.config.SHUFFLE = False
        self.config.NUM_WORKERS = 8

    def get(self):
        return self.config


class AnalyzeLoaderConfig(Config):

    def get(self):
        self.config.BATCH_SIZE = 1
        self.config.SHUFFLE = False
        self.config.NUM_WORKERS = 0
        return self.config


class DebugLoaderConfig(Config):

    def get(self):
        self.config.BATCH_SIZE = 4
        self.config.SHUFFLE = True
        self.config.NUM_WORKERS = 0
        return self.config


"""
Model configs
"""


class ModelConfig(Config):

    def get(self):
        self.config.GRID_SIZE = 8
        self.config.DESCRIPTOR_SIZE = 32
        self.config.NMS_KERNEL_SIZE = 15
        return self.config


class DebugModelConfig(Config):

    def get(self):
        self.config.GRID_SIZE = 8
        self.config.DESCRIPTOR_SIZE = 4
        self.config.NMS_KERNEL_SIZE = 15
        return self.config


"""
Criterion configs
"""


class CriterionConfig(Config):

    def get(self):
        self.config.DES_LAMBDA = 1
        self.config.MARGIN = 1

        self.config.DET_LAMBDA = 120

        self.config.NMS_THRESH = 0.0
        self.config.NMS_K_SIZE = 5

        self.config.TOP_K = 512

        self.config.GAUSS_K_SIZE = 15
        self.config.GAUSS_SIGMA = 0.5
        return self.config


"""
Metric config
"""


class MetricConfig(Config):

    def get(self):
        self.config.DET_THRESH = 5.0
        return self.config


"""
Log config
"""


class TrainLogConfig(Config):

    def get(self):
        self.config.LOSS_LOG_INTERVAL = 2
        self.config.METRIC_LOG_INTERVAL = 8
        return self.config


class ValLogConfig(Config):

    def get(self):
        self.config.LOG_INTERVAL = 20
        return self.config


class ShowLogConfig(Config):

    def get(self):
        self.config.LOG_INTERVAL = 5
        return self.config


class DebugLogConfig(Config):

    def get(self):
        self.config.LOSS_LOG_INTERVAL = 1
        self.config.METRIC_LOG_INTERVAL = 1
        self.config.LOG_INTERVAL = 1
        return self.config


"""
Experiment config
"""


class TrainExperimentConfig(Config):

    def get(self):
        self.config.NUM_EPOCHS = 3000
        self.config.SAVE_INTERVAL = 1000
        self.config.N_SAVED = 3
        return self.config


class VSATExperimentConfig(Config):

    def get(self):
        self.config.NUM_EPOCHS = 1
        return self.config


class DebugExperimentConfig(Config):

    def get(self):
        self.config.NUM_EPOCHS = 100
        self.config.SAVE_INTERVAL = 1
        self.config.N_SAVED = 3
        return self.config
