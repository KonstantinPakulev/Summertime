from easydict import EasyDict

from Net.experiments.main_experiment import *
from Net.source.nn.criterion import HardQuadTripletSOSRLoss


class TrainExperimentAlter(TrainExperiment):

    def get_criterion_settings(self):
        cs = EasyDict()

        cs.DES_LAMBDA = 1
        cs.MARGIN = 1
        cs.NUM_NEG = 64
        cs.SOS_NEG = 32

        cs.DET_LAMBDA = 100000

        cs.NMS_THRESH = 0.0
        cs.NMS_K_SIZE = 5

        cs.TOP_K = 512

        cs.GAUSS_K_SIZE = 15
        cs.GAUSS_SIGMA = 0.5

        return cs


class TrainExperimentMax(TrainExperiment):

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
        ms.DESCRIPTOR_SIZE = 128

        return ms


class DebugTrainExperiment(TrainExperiment):

    def get_dataset_settings(self):
        ds = EasyDict()

        ds.DATASET_ROOT = "../data/hpatch_v_sequence"
        ds.DATASET_MEAN = 0.4230204841414801
        ds.DATASET_STD = 0.25000138349993173
        ds.TRAIN_CSV = "debug.csv"
        ds.VAL_CSV = "debug.csv"
        ds.SHOW_CSV = "debug.csv"

        return ds

    def get_loaders_settings(self):
        ls = EasyDict()

        ls.TRAIN_BATCH_SIZE = 1
        ls.VAL_BATCH_SIZE = 1
        ls.SHOW_BATCH_SIZE = 1
        ls.NUM_WORKERS = 0

        return ls

    def get_model_settings(self):
        ms = EasyDict()

        ms.GRID_SIZE = 8
        ms.DESCRIPTOR_SIZE = 4

        return ms

    def get_log_settings(self):
        ls = EasyDict()

        ls.TRAIN = EasyDict()
        ls.TRAIN.LOSS_LOG_INTERVAL = 1
        ls.TRAIN.METRIC_LOG_INTERVAL = 1

        ls.VAL = EasyDict()
        ls.VAL.LOG_INTERVAL = 1

        ls.SHOW = EasyDict()
        ls.SHOW.LOG_INTERVAL = 1

        return ls

    def get_experiment_settings(self):
        es = EasyDict()

        es.NUM_EPOCHS = 300

        return es

    def get_checkpoint_settings(self):
        cs = EasyDict()

        cs.SAVE_INTERVAL = 1
        cs.N_SAVED = 3

        return cs
