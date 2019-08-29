from Net._legacy.experiments.main_experiment import *

from Net.source.nn.criterion import DenseInterQTripletLoss
from Net._legacy.source.nn.main_alter_criterion import *


class TEAlter(TE):

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = DenseInterQTripletLoss(ms.GRID_SIZE, cs.MARGIN)


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
        self.criterions[DES_CRITERION] = HardQuadCenterRadiusTripletLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG,
                                                                         cs.DES_LAMBDA)


class TED(TE):

    def init_config(self):
        self.config = DetectorConfig()

    def init_models(self):
        ms = self.config.model
        self.models[MODEL] = NetVGG(ms.GRID_SIZE, ms.DESCRIPTOR_SIZE, ms.NMS_KERNEL_SIZE, True).to(self.device)

    def iteration(self, engine, batch):
        model = self.models[MODEL]

        mse_criterion = self.criterions[DET_CRITERION]

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, _, _ = model(image1)
        score2, _, _ = model(image2)

        det_loss1, kp1 = mse_criterion(score1, score2, homo12)
        det_loss2, kp2 = mse_criterion(score2, score1, homo21)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        det_loss = (det_loss1 + det_loss2) / 2

        return {
            LOSS: det_loss,

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP2_MASK: wv_kp2_mask,
            WV_KP1_MASK: wv_kp1_mask
        }

    def inference(self, engine, batch):
        model = self.models[MODEL]

        ls = self.config.criterion

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, _, _ = model(image1)
        score2, _, _ = model(image2)

        _, kp1 = prepare_gt_score(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
        _, kp2 = prepare_gt_score(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        return {
            S_IMAGE1: batch[S_IMAGE1],
            S_IMAGE2: batch[S_IMAGE2],

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP2_MASK: wv_kp2_mask,
            WV_KP1_MASK: wv_kp1_mask
        }

    def bind_events(self):
        TrainDetMetricBinder(self.config, self.writer).bind(self.engines, self.loaders)

    def analyze_inference(self):
        self.init_config()

        ds = self.config.dataset
        ls = self.config.loader

        analyze_transform = [GrayscaleOld(),
                             NormalizeOld(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                             RescaleOld((960, 1280)),
                             RescaleOld((320, 640)),
                             ToTensorOld()]

        analyze_loader = DataLoader(
            TE.get_dataset(os.path.join("../", ds.DATASET_ROOT),
                           ds.ANALYZE_CSV, analyze_transform, True),
            batch_size=ls.ANALYZE_BATCH_SIZE)

        self.init_models()

        self.load_checkpoint()

        self.models[MODEL].eval()

        with torch.no_grad():
            batch = analyze_loader.__iter__().__next__()
            model = self.models[MODEL]

            ls = self.config.criterion

            image1, image2, homo12, homo21 = (
                batch[IMAGE1].to(self.device),
                batch[IMAGE2].to(self.device),
                batch[HOMO12].to(self.device),
                batch[HOMO21].to(self.device)
            )

            score1, _, debug1 = model(image1)
            score2, _, debug2 = model(image2)

            _, kp1 = prepare_gt_score(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
            _, kp2 = prepare_gt_score(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

            w_kp1 = warp_points(kp1, homo12)
            w_kp2 = warp_points(kp2, homo21)

            wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
            wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

            output = {
                S_IMAGE1: batch[S_IMAGE1],
                S_IMAGE2: batch[S_IMAGE2],

                DEBUG1: debug1,
                DEBUG2: debug2,

                HOMO12: homo12,
                HOMO21: homo21,

                SCORE1: score1,
                SCORE2: score2,

                KP1: kp1,
                KP2: kp2,

                W_KP1: w_kp1,
                W_KP2: w_kp2,

                WV_KP2_MASK: wv_kp2_mask,
                WV_KP1_MASK: wv_kp1_mask
            }

        return output


class TrainInterLossExperiment(TrainExperiment):

    def get_criterions(self, models_configs, criterions_configs):
        det_criterion = MSELoss(criterions_configs.NMS_THRESH, criterions_configs.NMS_K_SIZE,
                                criterions_configs.TOP_K,
                                criterions_configs.GAUSS_K_SIZE, criterions_configs.GAUSS_SIGMA,
                                criterions_configs.DET_LAMBDA)
        des_criterion = DenseInterQTripletLoss(models_configs.GRID_SIZE, criterions_configs.MARGIN,
                                               criterions_configs.DES_LAMBDA)
        return det_criterion, des_criterion


class TrainTriLossExperiment(TrainExperiment):

    def get_criterions(self, models_configs, criterions_configs):
        det_criterion = MSELoss(criterions_configs.NMS_THRESH, criterions_configs.NMS_K_SIZE,
                                criterions_configs.TOP_K,
                                criterions_configs.GAUSS_K_SIZE, criterions_configs.GAUSS_SIGMA,
                                criterions_configs.DET_LAMBDA)
        des_criterion = DenseInterTripletLoss(models_configs.GRID_SIZE, criterions_configs.MARGIN,
                                              criterions_configs.DES_LAMBDA)
        return det_criterion, des_criterion
