from easydict import EasyDict

from Net.experiments.main_experiment import *
from Net.source.nn.criterion import HardQuadTripletLoss, HardQuadTripletSOSRLoss


class TrainExperimentAlter(TrainExperiment):

    def get_criterion_settings(self):
        cs = EasyDict()

        cs.DES_LAMBDA = 1
        cs.MARGIN = 1
        cs.NUM_NEIGH = 25
        cs.NUM_NEG = 1

        cs.DET_LAMBDA = 100000

        cs.NMS_THRESH = 0.0
        cs.NMS_K_SIZE = 5

        cs.TOP_K = 512

        cs.GAUSS_K_SIZE = 15
        cs.GAUSS_SIGMA = 0.5

        return cs


class TrainExperimentQHT(TrainExperimentAlter):

    def init_criterions(self):
        ms = self.get_model_settings()
        cs = self.get_criterion_settings()

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = HardQuadTripletLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEIGH, cs.NUM_NEG,
                                                             cs.DES_LAMBDA)


class TrainExperimentSOSR(TrainExperiment):

    def get_criterion_settings(self):
        cs = EasyDict()

        cs.DES_LAMBDA = 1
        cs.MARGIN = 1
        cs.NUM_NEIGH = 25
        cs.NUM_NEG = 1
        cs.SOS_NEG = 4

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
        self.criterions[DES_CRITERION] = HardQuadTripletSOSRLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEIGH, cs.NUM_NEG,
                                                                 cs.SOS_NEG,
                                                                 cs.DES_LAMBDA)

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

        w_kp1 = warp_keypoints(kp1, homo12)
        w_kp2 = warp_keypoints(kp2, homo21)

        des_loss1 = triplet_criterion(kp1, w_kp1, kp1_desc, desc2)
        des_loss2 = triplet_criterion(kp2, w_kp2, kp2_desc, desc1)

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


class DebugTrainExperiment(TrainExperimentSOSR):

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
