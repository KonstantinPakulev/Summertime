from Net._legacy.source.nn.rf_criterion import MSERFLoss, MSEDiffRFLoss


class TEDRF(TED):

    def init_criterions(self):
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSERFLoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                   cs.TOP_K,
                                                   cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)


class TEDDiffRF(TED):

    def init_criterions(self):
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSEDiffRFLoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                       cs.TOP_K,
                                                       cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
