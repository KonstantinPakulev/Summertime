import torch

import Net.source.nn.net.utils.endpoint_utils as f
import Net.source.datasets.dataset_utils as d
import Net.source.utils.metric_utils as meu

from Net.source.utils.metric_utils import repeatability_score, match_score, mean_matching_accuracy, \
    epipolar_match_score, relative_pose_error, relative_param_pose_error

"""
Batch and endpoint transformers
"""


class RepTransformer:

    def __init__(self, px_thresh, detailed=False):
        self.px_thresh = px_thresh
        self.detailed = detailed

    def __call__(self, output):
        batch, endpoint = output

        kp1, kp2 = endpoint[f.KP1], endpoint[f.KP2]
        w_kp1, w_kp2 = endpoint[f.W_KP1], endpoint[f.W_KP2]
        w_vis_kp1_mask, w_vis_kp2_mask = endpoint[f.W_VIS_KP1_MASK], endpoint[f.W_VIS_KP2_MASK]

        metric = repeatability_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                                     self.px_thresh, self.detailed)

        if self.detailed:
            num_thresh = len(self.px_thresh)

            rep_scores, num_matches, num_vis_gt_matches, _, _ = metric
            num_vis_gt_matches = num_vis_gt_matches.unsqueeze(0).repeat(num_thresh, 1)

            detailed_rep = prepare_detailed_metric({meu.REP: rep_scores,
                                                    meu.REP_NUM_MATCHES: num_matches,
                                                    meu.REP_NUM_VIS_GT_MATCHES: num_vis_gt_matches},
                                                   batch, num_thresh)

            return detailed_rep

        else:
            return metric


class MSTransformer:

    def __init__(self, px_thresh, dd_measure, detailed=False):
        self.px_thresh = px_thresh
        self.dd_measure = dd_measure
        self.detailed = detailed

    def __call__(self, output):
        batch, endpoint = output

        kp1, kp2 = endpoint[f.KP1], endpoint[f.KP2]
        w_kp1, w_kp2 = endpoint[f.W_KP1], endpoint[f.W_KP2]
        w_vis_kp1_mask, w_vis_kp2_mask = endpoint[f.W_VIS_KP1_MASK], endpoint[f.W_VIS_KP2_MASK]
        kp1_desc, kp2_desc = endpoint[f.KP1_DESC], endpoint[f.KP2_DESC]

        metric = match_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, kp1_desc, kp2_desc,
                             self.px_thresh, self.dd_measure, self.detailed)

        if self.detailed:
            num_thresh = len(self.px_thresh)

            match_scores, num_matches, num_vis_gt_matches, _, _ = metric
            num_vis_gt_matches = num_vis_gt_matches.unsqueeze(0).repeat(num_thresh, 1)

            detailed_ms = prepare_detailed_metric({meu.MS: match_scores,
                                                   meu.MS_NUM_MATCHES: num_matches,
                                                   meu.MS_NUM_VIS_GT_MATCHES: num_vis_gt_matches},
                                                  batch, num_thresh)

            return detailed_ms

        else:
            return metric


class MMATransformer:

    def __init__(self, px_thresh, dd_measure, detailed=False):
        self.px_thresh = px_thresh
        self.dd_measure = dd_measure
        self.detailed = detailed

    def __call__(self, output):
        batch, endpoint = output

        kp1, kp2 = endpoint[f.KP1], endpoint[f.KP2]
        w_kp1, w_kp2 = endpoint[f.W_KP1], endpoint[f.W_KP2]
        w_vis_kp1_mask, w_vis_kp2_mask = endpoint[f.W_VIS_KP1_MASK], endpoint[f.W_VIS_KP2_MASK]
        kp1_desc, kp2_desc = endpoint[f.KP1_DESC], endpoint[f.KP2_DESC]

        metric = mean_matching_accuracy(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, kp1_desc, kp2_desc,
                                        self.px_thresh, self.dd_measure, self.detailed)

        if self.detailed:
            num_thresh = len(self.px_thresh)

            mma_scores, num_matches, num_vis_gt_matches, _, _ = metric
            num_vis_gt_matches = num_vis_gt_matches.unsqueeze(0).repeat(num_thresh, 1)

            detailed_mma = prepare_detailed_metric({meu.MMA: mma_scores,
                                                    meu.MMA_NUM_MATCHES: num_matches,
                                                    meu.MMA_NUM_VIS_GT_MATCHES: num_vis_gt_matches},
                                                   batch, num_thresh)

            return detailed_mma

        else:
            return metric


class EMSTransformer:

    def __init__(self, px_thresh, device, detailed=False):
        self.px_thresh = px_thresh
        self.device = device
        self.detailed = detailed

    def __call__(self, output):
        batch, endpoint = output

        kp1, kp2 = endpoint[f.KP1], endpoint[f.KP2]
        w_kp1, w_kp2 = endpoint[f.W_KP1], endpoint[f.W_KP2]
        w_vis_kp1_mask, w_vis_kp2_mask = endpoint[f.W_VIS_KP1_MASK], endpoint[f.W_VIS_KP2_MASK]
        kp1_desc, kp2_desc = endpoint[f.KP1_DESC], endpoint[f.KP2_DESC]

        shift_scale1, shift_scale2 = batch[d.SHIFT_SCALE1].to(self.device), batch[d.SHIFT_SCALE2].to(self.device)
        intrinsics1, intrinsics2 = batch[d.INTRINSICS1].to(self.device), batch[d.INTRINSICS2].to(self.device)
        extrinsics1, extrinsics2 = batch[d.EXTRINSICS1].to(self.device), batch[d.EXTRINSICS2].to(self.device)

        metric = epipolar_match_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                                      kp1_desc, kp2_desc, shift_scale1, shift_scale2,
                                      intrinsics1, intrinsics2, extrinsics1, extrinsics2,
                                      self.px_thresh, None, self.detailed)

        if self.detailed:
            num_thresh = len(self.px_thresh)

            em_scores, num_matches, num_vis_gt_matches, _, _ = metric
            num_vis_gt_matches = num_vis_gt_matches.unsqueeze(0).repeat(num_thresh, 1)

            detailed_ems = prepare_detailed_metric({meu.EMS: em_scores,
                                                    meu.EMS_NUM_MATCHES: num_matches,
                                                    meu.EMS_NUM_VIS_GT_MATCHES: num_vis_gt_matches},
                                                   batch, num_thresh)

            return detailed_ems

        else:
            return metric


class PoseTransformer:

    def __init__(self, px_thresh, device):
        self.px_thresh = px_thresh
        self.device = device

    def __call__(self, output):
        batch, endpoint = output

        kp1, kp2 = endpoint[f.KP1], endpoint[f.KP2]
        kp1_desc, kp2_desc = endpoint[f.KP1_DESC], endpoint[f.KP2_DESC]

        shift_scale1, shift_scale2 = batch[d.SHIFT_SCALE1].to(self.device), batch[d.SHIFT_SCALE2].to(self.device)
        intrinsics1, intrinsics2 = batch[d.INTRINSICS1].to(self.device), batch[d.INTRINSICS2].to(self.device)
        extrinsics1, extrinsics2 = batch[d.EXTRINSICS1].to(self.device), batch[d.EXTRINSICS2].to(self.device)

        R_err, t_err, est_inl_mask = relative_pose_error(kp1, kp2, kp1_desc, kp2_desc, shift_scale1, shift_scale2,
                                                         intrinsics1, intrinsics2, extrinsics1, extrinsics2,
                                                         self.px_thresh)
        num_thresh = len(self.px_thresh)

        detailed_pose = prepare_detailed_metric({meu.R_ERR: R_err,
                                                 meu.T_ERR: t_err,
                                                 meu.NUM_INL: est_inl_mask.sum(dim=-1)}, batch, num_thresh)

        return detailed_pose


class ParamPoseTransformer:

    def __init__(self, px_thresh, device):
        self.px_thresh = px_thresh
        self.device = device

    def __call__(self, output):
        batch, endpoint = output

        kp1, kp2 = endpoint[f.KP1], endpoint[f.KP2]
        kp1_desc, kp2_desc = endpoint[f.KP1_DESC], endpoint[f.KP2_DESC]

        shift_scale1, shift_scale2 = batch[d.SHIFT_SCALE1].to(self.device), batch[d.SHIFT_SCALE2].to(self.device)
        intrinsics1, intrinsics2 = batch[d.INTRINSICS1].to(self.device), batch[d.INTRINSICS2].to(self.device)
        extrinsics1, extrinsics2 = batch[d.EXTRINSICS1].to(self.device), batch[d.EXTRINSICS2].to(self.device)

        R_param_err, t_param_err, success_mask = relative_param_pose_error(kp1, kp2, kp1_desc, kp2_desc, shift_scale1, shift_scale2,
                                                                           intrinsics1, intrinsics2, extrinsics1, extrinsics2, self.px_thresh)
        num_thresh = len(self.px_thresh)

        detailed_param_pose = prepare_detailed_metric({meu.R_PARAM_ERR: R_param_err,
                                                       meu.T_PARAM_ERR: t_param_err,
                                                       meu.SUCCESS_MASK: success_mask}, batch, num_thresh)

        return detailed_param_pose





"""
Support utils
"""


def prepare_detailed_metric(data, batch, num_thresh):
    detailed_metric = [{} for _ in range(num_thresh)]

    for i in range(num_thresh):
        detailed_metric[i][d.SCENE_NAME] = batch.get(d.SCENE_NAME)

        detailed_metric[i][d.IMAGE1_NAME] = batch.get(d.IMAGE1_NAME)
        detailed_metric[i][d.IMAGE2_NAME] = batch.get(d.IMAGE2_NAME)

        detailed_metric[i][d.ID1] = batch.get(d.ID1)
        detailed_metric[i][d.ID2] = batch.get(d.ID2)

        for key, value in data.items():
            detailed_metric[i][key] = value[i]

    return detailed_metric


# Legacy code

# class TimeTransformer:
#
#     def __call__(self, output):
#         _, endpoint = output
#
#     return (endpoint[f.MODEL_INFO1][a.FORWARD_TIME] + endpoint[f.MODEL_INFO2][a.FORWARD_TIME]) / 2


# class DescriptorAnalysisTransformer:
#
#     def __init__(self, grid_size):
#         self.grid_size = grid_size
#
#     def __call__(self, output):
#         batch, endpoint = output
#
#         image1, image2 = batch.get(d.IMAGE1), batch.get(d.IMAGE2)
#         desc1, desc2 = endpoint[f.DESC1], endpoint[f.DESC2]
#
#         w_desc_grid1, w_vis_desc_grid_mask1, \
#         w_desc_grid2, w_vis_desc_grid_mask2 = create_w_desc_grid(image1, image2, desc1, desc2, batch, self.grid_size)
#
#         pos_dist1, neg_dist1, pos_second_dist1, neg_second_dist1, pos_num1, vis_num1 = \
#             measure_pos_neg_desc_dist(desc1, desc2, w_desc_grid1, w_vis_desc_grid_mask1, self.grid_size)
#
#         pos_dist2, neg_dist2, pos_second_dist2, neg_second_dist2, pos_num2, vis_num2 = \
#             measure_pos_neg_desc_dist(desc2, desc1, w_desc_grid2, w_vis_desc_grid_mask2, self.grid_size)
#
#         pos_dist = (pos_dist1 + pos_dist2) / 2
#         neg_dist = (neg_dist1 + neg_dist2) / 2
#         pos_second_dist = (pos_second_dist1 + pos_second_dist2) / 2
#         neg_second_dist = (neg_second_dist1 + neg_second_dist2) / 2
#         pos_num = (pos_num1 + pos_num2) / 2
#         vis_num = (vis_num1 + vis_num2) / 2
#
#         return [{an.POS_DESC_DIST: pos_dist.cpu(),
#                 an.NEG_DESC_DIST: neg_dist.cpu(),
#                 an.POS_DESC_DIST2: pos_second_dist.cpu(),
#                 an.NEG_DESC_DIST2: neg_second_dist.cpu(),
#                 an.POS_NUM: pos_num,
#                 an.VIS_NUM: vis_num}]
# if self.gt:
#     est_rel_pose, est_inl_mask, matches_mask, nn_ids1 = \
#         estimate_rel_poses_gt_opengv(kp1, endpoint[f.W_KP1], kp2, endpoint[f.W_KP2],
#                                      endpoint[f.W_VIS_KP1_MASK], endpoint[f.W_VIS_KP2_MASK],
#                                      intrinsics1, intrinsics2,
#                                      shift_scale1, shift_scale2,
#                                      self.px_thresh, True)
# est_rel_pose, matches_mask, nn_ids1 = \
#     estimate_rel_poses_opencv(kp1, kp2, kp1_desc, kp2_desc,
#                               intrinsics1, intrinsics2,
#                               shift_scale1, shift_scale2,
#                               self.px_thresh, self.dd_measure, True)

# num_ep_inl = torch.zeros(kp1.shape[0], dtype=torch.long)
#
# for b in range(kp1.shape[0]):
#     b_mask = est_inl_mask[i, b]
#     ep_dist = epipolar_distance(r_kp1[b, b_mask], nn_o_kp2[b, b_mask], F_gt)
#     num_ep_inl[b] = (ep_dist < self.ep_thresh).sum()
#
# detailed_pose[i][ev.NUM_EP_INL] = num_ep_inl
