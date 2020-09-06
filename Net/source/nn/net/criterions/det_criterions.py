import torch
from torch import nn as nn
from torch.nn import functional as F

import Net.source.core.experiment as exp

from Net.source.nn.net.utils.criterion_utils import prepare_gt_score, gaussian_filter


# TODO. One of the solution to the different scale detection problem may be optimizing MSE between patches
#  - not the score maps. Try it if pose loss wouldn't work

class DetTopKLoss(nn.Module):

    @staticmethod
    def from_config(criterion_config):
        detector_config = criterion_config[exp.DET]
        return DetTopKLoss(detector_config[exp.NMS_KERNEL_SIZE], detector_config[exp.TOP_K],
                           detector_config[exp.GAUSS_KERNEL_SIZE], detector_config[exp.GAUSS_SIGMA],
                           detector_config[exp.LAMBDA])

    def __init__(self, nms_kernel_size, top_k, gauss_kernel_size, gauss_sigma, loss_lambda):
        super().__init__()
        self.nms_kernel_size = nms_kernel_size
        self.top_k = top_k

        self.gauss_kernel_size = gauss_kernel_size
        self.gauss_sigma = gauss_sigma

        self.loss_lambda = loss_lambda

    def forward(self, score1, w_score2, w_vis_mask2):
        """
        :param score1: B x 1 x H x W
        :param w_score2: B x 1 x H x W
        :param w_vis_mask2: B x 1 x H x W
        """
        b = score1.size(0)

        gt_score1 = prepare_gt_score(w_score2, self.nms_kernel_size, self.top_k, w_vis_mask2).detach()
        gt_score1 = gaussian_filter(gt_score1, self.gauss_kernel_size, self.gauss_sigma)

        loss = F.mse_loss(score1, gt_score1, reduction='none') * w_vis_mask2.float()  # B x 1 x H x W
        loss = loss.view(b, -1).sum(dim=-1) / w_vis_mask2.float().view(b, -1).sum(dim=-1).clamp(min=1e-8)  # B
        loss = self.loss_lambda * loss.mean()

        return loss


class DetConfLoss(nn.Module):

    @staticmethod
    def from_config(criterion_config):
        detector_config = criterion_config[exp.DET_CONF]
        return DetConfLoss(detector_config[exp.LAMBDA])

    def __init__(self, loss_lambda):
        super().__init__()
        self.loss_lambda = loss_lambda

    def forward(self, log_conf_score1, pos_sim1, neg_sim1):
        """
        :param log_conf_score1: B x 1 x cH x cW
        :param pos_sim1: B x cH * cW
        :param neg_sim1: B x cH * cW
        """
        b = log_conf_score1.size(0)

        log_softmax_conf_score = log_conf_score1.view(b, -1).log_softmax(dim=-1)
        log_softmax_conf_gt = (neg_sim1 - pos_sim1).log_softmax(dim=-1).detach()

        loss = self.loss_lambda * F.mse_loss(log_softmax_conf_score, log_softmax_conf_gt)

        return loss


# Legacy code

# log_softmax_conf_score = log_conf_score1.view(b, -1).log_softmax(dim=-1)
# log_softmax_conf = (neg_sim1 - pos_sim1).log_softmax(dim=-1).detach()
# loss = F.mse_loss(log_softmax_conf_score, log_softmax_conf, reduction='none') * w_vis_desc_grid_mask1.float()  # B x cH * cW
# loss = loss.sum(dim=-1) / w_vis_desc_grid_mask1.float().sum(dim=-1).clamp(min=1e-8)
# loss = self.loss_lambda * loss.mean()
#
#         if self.loss_version == '1':
#             gt_score1 = prepare_gt_score_old(w_score2, self.nms_kernel_size, self.top_k).detach()
#
#         elif self.loss_version == '2':