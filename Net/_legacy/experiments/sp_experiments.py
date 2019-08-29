from Net._legacy.experiments.main_experiment import *
from Net.experiments.config import SPConfig
from Net._legacy.source.nn.sp_criterion import PairHingeLoss, DenseTripletLoss, DenseInterTripletLoss


class TESP(TE):

    def init_config(self):
        self.config = SPConfig()

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = PairHingeLoss(ms.GRID_SIZE, cs.POS_LAMBDA, cs.POS_MARGIN, cs.NEG_MARGIN)

    def iteration(self, engine, batch):
        model = self.models[MODEL]

        mse_criterion = self.criterions[DET_CRITERION]
        hinge_criterion = self.criterions[DES_CRITERION]

        ms = self.config.model

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

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        des_loss1 = hinge_criterion(desc1, desc2, homo12)
        des_loss2 = hinge_criterion(desc2, desc1, homo21)

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

            WV_KP2_MASK: wv_kp2_mask,
            WV_KP1_MASK: wv_kp1_mask,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc
        }


class TESPTriplet(TE):

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = DenseQTripletLoss(ms.GRID_SIZE, cs.MARGIN)

    def iteration(self, engine, batch):
        model = self.models[MODEL]

        mse_criterion = self.criterions[DET_CRITERION]
        triple_criterion = self.criterions[DES_CRITERION]

        ms = self.config.model

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

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        des_loss1 = triple_criterion(score1, score2, desc1, desc2, homo12, homo21)
        des_loss2 = triple_criterion(score2, score1, desc2, desc1, homo21, homo12)

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

            WV_KP2_MASK: wv_kp2_mask,
            WV_KP1_MASK: wv_kp1_mask,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc
        }


class TESPTriInt(TESPTriplet):

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = DenseInterTripletLoss(ms.GRID_SIZE, cs.MARGIN)

