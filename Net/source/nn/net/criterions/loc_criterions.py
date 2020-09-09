import torch
import torch.nn as nn

import Net.source.core.experiment as exp

from Net.source.utils.math_utils import revert_data_transform, epipolar_distance, compose_gt_transform, \
    E_param, change_intrinsics
from Net.source.utils.matching_utils import select_kp, get_gt_matches

from Net.source.utils.pose_utils import ParamRelPose, ParamTruncRelPose


class EpipolarLoss(nn.Module):

    @staticmethod
    def from_config(criterion_config):
        ep_config = criterion_config[exp.EP]
        return EpipolarLoss(ep_config[exp.PX_THRESH], ep_config[exp.LAMBDA])

    def __init__(self, px_thresh, loss_lambda):
        """
        :type px_thresh: float
        """
        super().__init__()
        self.px_thresh = px_thresh

        self.loss_lambda = loss_lambda

    def forward(self, kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                shift_scale1, shift_scale2, intrinsics1, intrinsics2, extrinsics1, extrinsics2):
        mutual_gt_matches_mask, nn_kp_ids = \
            get_gt_matches(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, self.px_thresh)

        r_kp1 = revert_data_transform(kp1, shift_scale1)
        r_kp2 = revert_data_transform(kp2, shift_scale2)

        nn_r_kp2 = select_kp(r_kp2, nn_kp_ids)

        F = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2)

        # TODO. Squared epipolar distance maybe?

        ep_dist = epipolar_distance(r_kp1, nn_r_kp2, F).clamp(max=self.px_thresh)

        loss = ep_dist * mutual_gt_matches_mask.float()
        loss = loss.sum(dim=-1) / mutual_gt_matches_mask.float().sum(dim=-1).clamp(min=1e-8)
        loss = self.loss_lambda * loss.mean()

        return loss


class PoseLoss(nn.Module):

    def __init__(self, px_thresh, loss_lambda, loss_version):
        """
        :type px_thresh: float
        """
        super().__init__()
        self.px_thresh = px_thresh

        self.loss_lambda = loss_lambda

        self.loss_version = loss_version

    def forward(self, kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                shift_scale1, shift_scale2, intrinsics1, intrinsics2, extrinsics1, extrinsics2):
        mutual_gt_matches_mask, nn_kp_ids = \
            get_gt_matches(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, self.px_thresh)

        r_kp1 = revert_data_transform(kp1, shift_scale1)
        r_kp2 = revert_data_transform(kp2, shift_scale2)

        nn_r_kp2 = select_kp(r_kp2, nn_kp_ids)
        nn_r_i1_kp2 = change_intrinsics(nn_r_kp2, intrinsics2, intrinsics1)

        gt_E_param = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2, E_param)

        if self.loss_version == '1':
            t_px_thresh = torch.tensor(self.px_thresh).to(kp1.device)
            est_E_param, success_mask = ParamRelPose().apply(r_kp1, nn_r_i1_kp2, mutual_gt_matches_mask, intrinsics1, intrinsics2, t_px_thresh)

        elif self.loss_version == '2':
            t_px_thresh = torch.tensor(self.px_thresh).to(kp1.device)
            est_E_param, success_mask = ParamTruncRelPose().apply(r_kp1, nn_r_i1_kp2, nn_r_kp2, mutual_gt_matches_mask,
                                                                  intrinsics1, intrinsics2, extrinsics1, extrinsics2, t_px_thresh)

        else:
            raise NotImplementedError

        loss = (est_E_param - gt_E_param).norm(dim=-1)
        loss = loss.sum() / success_mask.float().sum().clamp(min=1e-8)
        loss = self.loss_lambda * loss

        return loss
