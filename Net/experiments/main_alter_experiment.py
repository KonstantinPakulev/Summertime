from Net.experiments.main_experiment import *
from Net.source.nn.main_alter_criterion import *


class TERSOSR(TE):

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = HardQuadTripletRadiusSOSRLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG, cs.SOS_NEG,
                                                                       cs.DES_LAMBDA)


class TENoSOSR(TE):

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = HardQuadTripletLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG, cs.DES_LAMBDA)

class TERFOSNoSOSR(TE):

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = HardQuadRadiusTripletLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG, cs.DES_LAMBDA)

class TECRFOSNoSOSR(TE):

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = HardQuadCenterRadiusTripletLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG, cs.DES_LAMBDA)
