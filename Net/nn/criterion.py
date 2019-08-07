import math
from bisect import bisect

import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.utils.image_utils import (create_coordinates_grid,
                                   warp_coordinates_grid,
                                   warp_keypoints,
                                   warp_image,
                                   dilate_filter,
                                   gaussian_filter,
                                   filter_border,
                                   select_keypoints)

from Net.utils.model_utils import sample_descriptors, space_to_depth


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
        # Filter border
        score2 = filter_border(score2)

        # Warp score2 to score1 space
        w_score2 = warp_image(score1, score2, homo12)

        # Create visibility mask of the first image
        vis_mask = torch.ones_like(score2).to(score2.device)
        vis_mask1 = warp_image(score1, vis_mask, homo12).gt(0).float()

        # Extract keypoints and prepare ground truth
        score1, kp1 = select_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)
        score1 = gaussian_filter(score1, self.gauss_k_size, self.gauss_sigma)

        gt_score1, _ = select_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        norm = vis_mask1.sum()
        loss = F.mse_loss(score1, gt_score1, reduction='none') * vis_mask1 / norm
        loss = loss.sum() * self.loss_lambda

        return loss, kp1


class HingeLoss(nn.Module):

    def __init__(self, grid_size,
                 pos_margin, neg_margin,
                 neg_samples,
                 loss_lambda):
        super().__init__()
        self.grid_size = grid_size

        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

        self.neg_samples = neg_samples

        self.loss_lambda = loss_lambda

    def forward(self, kp1, w_kp1, kp2, kp1_desc, kp2_desc, desc2):
        w_kp1_grid = w_kp1.float().unsqueeze(1)
        kp2_grid = kp2.float().unsqueeze(0)

        grid_dist = torch.norm(w_kp1_grid - kp2_grid, dim=-1)
        radius = math.sqrt(self.grid_size ** 2 / 2)
        neighbour_mask = (grid_dist <= 2 * radius + 0.1).float()

        positive = sample_descriptors(desc2, w_kp1, self.grid_size)

        # Cosine similarity measure
        desc_dot = torch.mm(kp1_desc, kp2_desc.t())
        desc_dot = desc_dot - neighbour_mask * 5

        positive_dot = (kp1_desc * positive).sum(dim=-1)
        negative_dot = desc_dot.topk(self.neg_samples, dim=-1)[0]

        balance_factor = self.neg_samples / 3.0
        loss = torch.clamp(self.pos_margin - positive_dot, min=0).sum() * balance_factor + \
               torch.clamp(negative_dot - self.neg_margin, min=0).sum()

        norm = self.loss_lambda / (kp1_desc.size(0) * self.neg_samples)
        loss = loss * norm

        return loss


class HardTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin

        self.loss_lambda = loss_lambda

    def forward(self, kp1, w_kp1, kp2, kp1_desc, kp2_desc, desc2):
        w_kp1_grid = w_kp1.float().unsqueeze(1)
        kp2_grid = kp2.float().unsqueeze(0)

        grid_dist = torch.norm(w_kp1_grid - kp2_grid, dim=-1)
        radius = math.sqrt(self.grid_size ** 2 / 2)
        neighbour_mask = (grid_dist <= 2 * radius + 0.1).float()

        anchor = kp1_desc.unsqueeze(1)
        positive = sample_descriptors(desc2, w_kp1, self.grid_size)
        negative = kp2_desc.unsqueeze(0)

        # L2 distance measure
        desc_dist = torch.norm(anchor - negative, dim=-1)
        desc_dist = desc_dist + neighbour_mask * 5

        positive_dist = torch.pairwise_distance(kp1_desc, positive)
        # Pick closest negative sample
        negative_dist = desc_dist.min(dim=-1)[0]

        loss = torch.clamp(positive_dist - negative_dist + self.margin, min=0).mean() * self.loss_lambda

        return loss
