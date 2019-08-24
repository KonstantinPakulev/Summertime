import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.source.utils.image_utils import (warp_image,
                                          gaussian_filter,
                                          filter_border,
                                          select_score_and_keypoints)


class MSERFLoss(nn.Module):

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

        score1 = filter_border(score1)
        w_score2 = filter_border(w_score2)

        # Create visibility mask of the first image
        w_vis_mask = warp_image(score1, torch.ones_like(score2).to(score2.device), homo12).gt(0).float()

        # Extract keypoints and prepare ground truth
        score1, kp1 = select_score_and_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)
        score1 = gaussian_filter(score1, self.gauss_k_size, self.gauss_sigma)

        gt_score1, _ = select_score_and_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        norm = w_vis_mask.sum()
        loss = F.mse_loss(score1, gt_score1, reduction='none') * w_vis_mask / norm
        loss = loss.sum() * self.loss_lambda

        return loss, kp1


class MSEDiffRFLoss(nn.Module):

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

        score1 = filter_border(score1)
        w_score2 = filter_border(w_score2)

        # Create visibility mask of the first image
        w_vis_mask = warp_image(score1, torch.ones_like(score2).to(score2.device), homo12).gt(0).float()

        # Extract keypoints and prepare ground truth
        _, kp1 = select_score_and_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)

        gt_score1, _ = select_score_and_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        norm = w_vis_mask.sum()
        loss = F.mse_loss(score1, gt_score1, reduction='none') * w_vis_mask / norm
        loss = loss.sum() * self.loss_lambda

        return loss, kp1