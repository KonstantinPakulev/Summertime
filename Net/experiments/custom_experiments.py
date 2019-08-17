from easydict import EasyDict

from Net.experiments.main_experiment import *
from Net.source.nn.criterion import HardQuadTripletSOSRLoss
from Net.source.nn.custom_criterion import HardTripletLoss


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


class TrainExperimentDet(TrainExperiment):

    def get_criterion_settings(self):
        cs = EasyDict()

        cs.DES_LAMBDA = 1
        cs.MARGIN = 1
        cs.NUM_NEG = 1

        cs.DET_LAMBDA = 100000

        cs.NMS_THRESH = 0.0
        cs.NMS_K_SIZE = 5

        cs.TOP_K = 512

        cs.GAUSS_K_SIZE = 15
        cs.GAUSS_SIGMA = 0.5

        return cs

    def init_criterions(self):
        ms = self.get_model_settings()
        cs = self.get_criterion_settings()

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = HardTripletLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG, cs.DES_LAMBDA)

    def init_engines(self):
        model = self.models[MODEL]

        optimizer = self.optimizers[OPTIMIZER]

        def train_iteration(engine, batch):
            model.train()

            with torch.autograd.set_detect_anomaly(True):
                endpoint = self.iteration(engine, batch)

                optimizer.zero_grad()
                endpoint[DET_LOSS].backward()
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



class DebugTrainExperiment(TrainExperimentDet):

    def get_dataset_settings(self):
        ds = EasyDict()

        ds.DATASET_ROOT = "../data/hpatch_v_sequence"
        ds.DATASET_MEAN = 0.4230204841414801
        ds.DATASET_STD = 0.25000138349993173
        ds.TRAIN_CSV = "debug.csv"
        ds.VAL_CSV = "debug.csv"
        ds.SHOW_CSV = "debug.csv"
        ds.ANALYZE_CSV = "analyze.csv"

        return ds

    def get_loaders_settings(self):
        ls = EasyDict()

        ls.TRAIN_BATCH_SIZE = 1
        ls.VAL_BATCH_SIZE = 1
        ls.SHOW_BATCH_SIZE = 1
        ls.ANALYZE_BATCH_SIZE = 1
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

        es.NUM_EPOCHS = 100

        return es

    def get_checkpoint_settings(self):
        cs = EasyDict()

        cs.SAVE_INTERVAL = 1
        cs.N_SAVED = 3

        return cs
