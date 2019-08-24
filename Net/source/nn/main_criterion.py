import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.source.utils.critetion_utils import calculate_interpolation_fos, calculate_interpolation_sos
from Net.source.utils.image_utils import (warp_image,
                                          filter_border,
                                          gaussian_filter,
                                          select_gt_and_keypoints)


class MSELoss(nn.Module):

    def __init__(self, nms_thresh, nms_k_size, top_k, gauss_k_size, gauss_sigma, loss_lambda):
        super().__init__()

        self.nms_thresh = nms_thresh
        self.nms_k_size = nms_k_size

        self.top_k = top_k

        self.gauss_k_size = gauss_k_size
        self.gauss_sigma = gauss_sigma

        self.loss_lambda = loss_lambda

    def forward(self, score1, score2, homo12):
        """
        :param score1: B x C x H x W
        :param score2: B x C x H x W
        :param homo12: B x 3 x 3
        :return: float, B x C x N x 2
        """
        # Warp score2 to score1 space
        w_score2 = warp_image(score1, score2, homo12)

        w_vis_mask = warp_image(score1, torch.ones_like(score2).to(score2.device), homo12).gt(0).float()

        _, kp1 = select_gt_and_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)

        gt_score1, _ = select_gt_and_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        loss = F.mse_loss(score1, gt_score1, reduction='none') * w_vis_mask
        loss = loss.sum() * self.loss_lambda / w_vis_mask.sum()

        return loss, kp1


class HardQuadTripletSOSRLoss(nn.Module):

    def __init__(self, grid_size, margin, num_neg, sos_neg, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin
        self.num_neg = num_neg

        self.sos_neg = sos_neg

        self.loss_lambda = loss_lambda


    def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
        """
        :param kp1 B x N x 2
        :param w_kp1: B x N x 2
        :param kp1_desc: B x N x C
        :param desc2: B x C x H x W
        :param homo12: B x 3 x 3
        :return: float
        """
        # Calculate FOS
        fos, w_kp1_desc, kp1_cells, kp1_w_cell_cell_ids, coo_grid = \
            calculate_interpolation_fos(kp1, w_kp1, kp1_desc, desc2, homo12, self.grid_size, self.margin, self.num_neg)

        # Calculate SOS
        sos = calculate_interpolation_sos(kp1_cells, kp1_w_cell_cell_ids, coo_grid, w_kp1, kp1_desc, w_kp1_desc, self.sos_neg)

        loss = (fos + sos) * self.loss_lambda

        return loss
