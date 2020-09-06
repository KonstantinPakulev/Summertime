import torch.nn.functional as F

import Net.source.core.experiment as exp
import Net.source.nn.net.utils.endpoint_utils as eu
import Net.source.nn.net.utils.criterion_utils as cu
import Net.source.datasets.dataset_utils as du

from Net.source.core.wrapper import AttachableModelWrapper
from Net.source.core.ignite_metrics import AveragePeriodicMetric, KeyTransformer

from Net.source.nn.net.criterions.det_criterions import DetTopKLoss, DetConfLoss
from Net.source.nn.net.criterions.desc_criterions import DescTripletLoss
from Net.source.nn.net.criterions.loc_criterions import PoseLoss, EpipolarLoss

from Net.source.nn.net.utils.criterion_utils import warp_image, create_w_desc_grid

POS_SIM1 = 'pos_sim1'
POS_SIM2 = 'pos_sim2'

NEG_SIM1 = 'neg_sim1'
NEG_SIM2 = 'neg_sim2'

W_DESC_GRID1 = 'w_desc_grid1'
W_DESC_GRID2 = 'w_desc_grid2'

W_VIS_DESC_GRID_MASK1 = 'w_vis_desc_grid_mask1'
W_VIS_DESC_GRID_MASK2 = 'w_vis_desc_grid_mask2'


class DetJointLossWrapper(AttachableModelWrapper):

    def __init__(self, device, criterion_config):
        super().__init__(device)
        self.topk_loss = DetTopKLoss.from_config(criterion_config)

    def forward(self, engine, batch, endpoint, bundle):
        image1, image2 = batch[du.IMAGE1].to(self.device), batch[du.IMAGE2].to(self.device)
        score1, score2 = endpoint[eu.SCORE1], endpoint[eu.SCORE2]

        # TODO. Measure scale factor from params for all new models

        w_score1, w_vis_mask1, w_score2, w_vis_mask2 = warp_image(score1, score2, batch, 0.5)

        det_topk_loss1 = self.topk_loss(score1, w_score2, w_vis_mask2)
        det_topk_loss2 = self.topk_loss(score2, w_score1, w_vis_mask1)

        det_topk_loss = (det_topk_loss1 + det_topk_loss2) / 2

        endpoint[cu.DET_LOSS] = det_topk_loss

        return det_topk_loss, endpoint, bundle

    def attach(self, engine, bundle):
        loss_log_iter = bundle.get(exp.LOSS_LOG_ITER)
        AveragePeriodicMetric(KeyTransformer(cu.DET_LOSS), loss_log_iter).attach(engine, cu.DET_LOSS)


class DetConfLossWrapper(AttachableModelWrapper):

    def __init__(self, device, criterion_config):
        super().__init__(device)
        self.conf_loss = DetConfLoss.from_config(criterion_config)

    def forward(self, engine, batch, endpoint, bundle):
        pos_sim1, pos_sim2 = bundle[POS_SIM1], bundle[POS_SIM2]
        neg_sim1, neg_sim2 = bundle[NEG_SIM1], bundle[NEG_SIM2]

        log_conf_score1, log_conf_score2 = endpoint[eu.LOG_CONF_SCORE1], endpoint[eu.LOG_CONF_SCORE2]

        det_conf_loss1 = self.conf_loss(log_conf_score1, pos_sim1, neg_sim1)
        det_conf_loss2 = self.conf_loss(log_conf_score2, pos_sim2, neg_sim2)

        det_conf_loss = (det_conf_loss1 + det_conf_loss2) / 2

        endpoint[cu.DET_CONF_LOSS] = det_conf_loss

        return det_conf_loss, endpoint, bundle

    def attach(self, engine, bundle):
        loss_log_iter = bundle.get(exp.LOSS_LOG_ITER)
        AveragePeriodicMetric(KeyTransformer(cu.DET_CONF_LOSS), loss_log_iter).attach(engine, cu.DET_CONF_LOSS)


class DescTripletLossWrapper(AttachableModelWrapper):

    def __init__(self, device, model_config, criterion_config):
        super().__init__(device)
        self.triplet_loss = DescTripletLoss.from_config(model_config, criterion_config)

        self.grid_size = model_config[exp.GRID_SIZE]

    def forward(self, engine, batch, endpoint, bundle):
        image1, image2 = batch[du.IMAGE1], batch[du.IMAGE2]

        w_desc_grid1, w_vis_desc_grid_mask1, w_desc_grid2, w_vis_desc_grid_mask2 = \
            create_w_desc_grid(image1.shape, image2.shape, batch, self.grid_size, self.device)

        desc1, desc2 = endpoint[eu.DESC1], endpoint[eu.DESC2]

        desc_loss1, pos_sim1, neg_sim1 = self.triplet_loss(desc1, desc2, w_desc_grid1, w_vis_desc_grid_mask1)
        desc_loss2, pos_sim2, neg_sim2 = self.triplet_loss(desc2, desc1, w_desc_grid2, w_vis_desc_grid_mask2)

        desc_loss = (desc_loss1 + desc_loss2) / 2

        endpoint[cu.DESC_LOSS] = desc_loss

        bundle[POS_SIM1] = pos_sim1
        bundle[NEG_SIM1] = neg_sim1

        bundle[POS_SIM2] = pos_sim2
        bundle[NEG_SIM2] = neg_sim2

        return desc_loss, endpoint, bundle

    def attach(self, engine, bundle):
        loss_log_iter = bundle.get(exp.LOSS_LOG_ITER)
        AveragePeriodicMetric(KeyTransformer(cu.DESC_LOSS), loss_log_iter).attach(engine, cu.DESC_LOSS)


class LocEpipolarLossWrapper(AttachableModelWrapper):

    def __init__(self, device, criterion_config):
        super().__init__(device)
        self.ep_loss = EpipolarLoss.from_config(criterion_config)


    def forward(self, engine, batch, endpoint, bundle):
        kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
        w_kp1, w_kp2 = endpoint[eu.W_KP1], endpoint[eu.W_KP2]
        w_vis_kp1_mask, w_vis_kp2_mask = endpoint[eu.W_VIS_KP1_MASK], endpoint[eu.W_VIS_KP2_MASK]

        shift_scale1, shift_scale2 = batch[du.SHIFT_SCALE1].to(self.device), batch[du.SHIFT_SCALE2].to(self.device)
        intrinsics1, intrinsics2 = batch[du.INTRINSICS1].to(self.device), batch[du.INTRINSICS2].to(self.device)
        extrinsics1, extrinsics2 = batch[du.EXTRINSICS1].to(self.device), batch[du.EXTRINSICS2].to(self.device)

        ep_loss1 = self.ep_loss(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, shift_scale1, shift_scale2,
                                intrinsics1, intrinsics2, extrinsics1, extrinsics2)

        ep_loss2 = self.ep_loss(kp2, kp1, w_kp2, w_kp1, w_vis_kp2_mask, w_vis_kp1_mask, shift_scale2, shift_scale1,
                                intrinsics2, intrinsics1, extrinsics2, extrinsics1)

        ep_loss = (ep_loss1 + ep_loss2) / 2

        endpoint[cu.EPIPOLAR_LOSS] = ep_loss

        return ep_loss, endpoint, bundle

    def attach(self, engine, bundle):
        loss_log_iter = bundle.get(exp.LOSS_LOG_ITER)
        AveragePeriodicMetric(KeyTransformer(cu.EPIPOLAR_LOSS), loss_log_iter).attach(engine, cu.EPIPOLAR_LOSS)


class LocPoseLossWrapper(AttachableModelWrapper):

    def __init__(self, device, criterion_config):
        super().__init__(device)
        config = criterion_config[exp.POSE]

        self.pose_loss = PoseLoss(config[exp.PX_THRESH], config[exp.LAMBDA], config[exp.LOSS_VERSION])

    def forward(self, engine, batch, endpoint, bundle):
        kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
        w_kp1, w_kp2 = endpoint[eu.W_KP1], endpoint[eu.W_KP2]
        w_vis_kp1_mask, w_vis_kp2_mask = endpoint[eu.W_VIS_KP1_MASK], endpoint[eu.W_VIS_KP2_MASK]

        shift_scale1, shift_scale2 = batch[du.SHIFT_SCALE1].to(self.device), batch[du.SHIFT_SCALE2].to(self.device)
        intrinsics1, intrinsics2 = batch[du.INTRINSICS1].to(self.device), batch[du.INTRINSICS2].to(self.device)
        extrinsics1, extrinsics2 = batch[du.EXTRINSICS1].to(self.device), batch[du.EXTRINSICS2].to(self.device)

        pose_loss = self.pose_loss(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                                   shift_scale1, shift_scale2, intrinsics1, intrinsics2, extrinsics1, extrinsics2)

        endpoint[cu.POSE_LOSS] = pose_loss

        return pose_loss, endpoint, bundle

    def attach(self, engine, bundle):
        loss_log_iter = bundle.get(exp.LOSS_LOG_ITER)
        AveragePeriodicMetric(KeyTransformer(cu.POSE_LOSS), loss_log_iter).attach(engine, cu.POSE_LOSS)


# Legacy wrappers
#
# class DetMSLossWrapper(AttachableModelWrapper):
#
#     def __init__(self, criterion_config):
#         super().__init__()
#         self.topk_loss = DetTopKLoss.from_config(criterion_config)
#
#     def forward(self, engine, batch, endpoint, bundle):
#         image1, image2 = batch.get(du.IMAGE1), batch.get(du.IMAGE2)
#         score1, score2 = endpoint[eu.SCORE1], endpoint[eu.SCORE2]
#
#         w_image1, w_image2 = warp_image(image1, image2, batch)
#         w_score1, w_score2 = warp_image(score1, score2, batch)
#
#         w_vis_mask1, w_vis_mask2 = w_image1.gt(0.0), w_image2.gt(0.0)
#
#         det_topk_loss1 = self.topk_loss(score1, w_score2, w_vis_mask2)
#         det_topk_loss2 = self.topk_loss(score2, w_score1, w_vis_mask1)
#
#         det_topk_loss = (det_topk_loss1 + det_topk_loss2) / 2
#
#         endpoint[cu.DET_LOSS] = det_topk_loss
#
#         return det_topk_loss, endpoint, bundle
#
#     def attach(self, engine, bundle):
#         loss_log_iter = bundle.get(exp.LOSS_LOG_ITER)
#         AveragePeriodicMetric(KeyTransformer(cu.DET_LOSS), loss_log_iter).attach(engine, cu.DET_LOSS)
