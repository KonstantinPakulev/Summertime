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

        return loss, kp1, vis_mask1


class ReceptiveHingeLoss(nn.Module):

    def __init__(self, grid_size,
                 pos_margin, neg_margin,
                 loss_lambda):
        super().__init__()
        self.grid_size = grid_size

        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

        self.loss_lambda = loss_lambda

    def forward(self, kp1, desc1, desc2, homo21, vis_mask1):
        """
        :param kp1: K x 4
        :param desc1: N x C x Hr x Wr
        :param desc2: N x C x Hr x Wr
        :param homo21: N x 3 x 3
        :param vis_mask1: Mask of the first image. N x 1 x H x W
        Note: 'r' suffix means reduced in 'grid_size' times
        """

        # Move keypoints coordinates to reduced spatial size
        kp_grid = kp1[:, [3, 2]].unsqueeze(0).float()
        kp_grid[:, 0] = kp_grid[:, 0] / self.grid_size
        kp_grid[:, 1] = kp_grid[:, 1] / self.grid_size

        # Warp reduced coordinate grid to desc1 viewpoint
        w_grid = create_coordinates_grid(desc2.size()) * self.grid_size + self.grid_size // 2
        w_grid = w_grid.type_as(desc2).to(desc2.device)
        w_grid = warp_coordinates_grid(w_grid, homo21)

        kp_grid = kp_grid.unsqueeze(2).unsqueeze(2)
        w_grid = w_grid.unsqueeze(1)

        n, _, hr, wr = desc1.size()

        # Reduce spatial dimensions of visibility mask
        vis_mask1 = space_to_depth(vis_mask1, self.grid_size).prod(dim=1)
        vis_mask1 = vis_mask1.reshape([n, 1, hr, wr])

        # Mask with homography induced correspondences
        grid_dist = torch.norm(kp_grid - w_grid, dim=-1)
        ones = torch.ones_like(grid_dist)
        zeros = torch.zeros_like(grid_dist)
        s = torch.where(grid_dist <= self.grid_size - 0.5, ones, zeros)

        ks1, ks2, ks3 = 31, 11, 5
        sigma1, sigma2, sigma3 = 7, 4, 2

        ns = s.clone().view(-1, 1, hr, wr)
        ns1 = gaussian_filter(ns, ks1, sigma1).view(n, kp1.size(0), hr, wr) - s
        ns2 = gaussian_filter(ns, ks2, sigma2).view(n, kp1.size(0), hr, wr) - s
        ns3 = gaussian_filter(ns, ks3, sigma3).view(n, kp1.size(0), hr, wr) - s
        ns = ns1 + ns2 + ns3

        # Apply visibility mask
        s *= vis_mask1
        ns *= vis_mask1

        # Sample descriptors
        kp1_desc = sample_descriptors(desc1, kp1, self.grid_size)

        # Calculate distance pairs
        s_kp1_desc = kp1_desc.permute(1, 0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
        s_desc2 = desc2.unsqueeze(2)
        dot_desc = torch.sum(s_kp1_desc * s_desc2, dim=1)

        pos_dist = (self.pos_margin - dot_desc).clamp(min=0)
        neg_dist = (dot_desc - self.neg_margin).clamp(min=0)

        neg_per_pos = 374.359
        pos_lambda = neg_per_pos * 0.3

        loss = pos_lambda * s * pos_dist + ns * neg_dist

        norm = kp1.size(0) * neg_per_pos
        loss = loss.sum() / norm * self.loss_lambda

        return loss, kp1_desc


class HardTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin

        self.loss_lambda = loss_lambda

    def forward(self, kp1, desc1, desc2, homo21, vis_mask1):
        """
        :param kp1: K x 4
        :param desc1: N x C x Hr x Wr
        :param desc2: N x C x Hr x Wr
        :param homo21: N x 3 x 3
        :param vis_mask1: Mask of the first image. N x 1 x H x W
        Note: 'r' suffix means reduced in 'grid_size' times
        """

        # Move keypoints coordinates to reduced spatial size
        kp_grid = kp1[:, [3, 2]].unsqueeze(0).float()
        kp_grid[:, 0] = kp_grid[:, 0] / self.grid_size
        kp_grid[:, 1] = kp_grid[:, 1] / self.grid_size

        # Warp reduced coordinate grid to desc1 viewpoint
        w_grid = create_coordinates_grid(desc2.size()) * self.grid_size + self.grid_size // 2
        w_grid = w_grid.type_as(desc2).to(desc2.device)
        w_grid = warp_coordinates_grid(w_grid, homo21)

        kp_grid = kp_grid.unsqueeze(2).unsqueeze(2)
        w_grid = w_grid.unsqueeze(1)

        n, _, hr, wr = desc1.size()

        # Reduce spatial dimensions of visibility mask
        vis_mask1 = space_to_depth(vis_mask1, self.grid_size).prod(dim=1)
        vis_mask1 = vis_mask1.reshape([n, 1, hr, wr])

        # Mask with homography induced correspondences
        grid_dist = torch.norm(kp_grid - w_grid, dim=-1)
        s = (grid_dist <= self.grid_size - 0.5).float()
        ns = 1 - s

        # Apply visibility mask
        s *= vis_mask1
        ns *= vis_mask1

        # Sample descriptors
        kp1_desc = sample_descriptors(desc1, kp1, self.grid_size)

        # Calculate distance pairs
        s_kp1_desc = kp1_desc.permute(1, 0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
        s_desc2 = desc2.unsqueeze(2)
        dist_desc = torch.norm(s_kp1_desc - s_desc2, dim=1)

        pos_dist = (dist_desc - ns * 5).view(n, kp1.size(0), -1).max(dim=-1)[0]
        neg_dist = (dist_desc + s * 5).view(n, kp1.size(0), -1).min(dim=-1)[0]

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0).mean() * self.loss_lambda

        return loss, kp1_desc
