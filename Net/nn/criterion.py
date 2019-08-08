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

from Net.utils.common_utils import kp2coord
from Net.utils.model_utils import sample_descriptors, space_to_depth
from Net.utils.math_utils import calculate_similarity_matrix, calculate_similarity_vector


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
        # TODO. Remove border filtering and replace it with eroding procedure.
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


class HardTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin

        self.loss_lambda = loss_lambda

    def forward(self, kp1, w_kp1, kp1_desc, desc2):
        # Calculate similarity measure between anchors and positives
        w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)

        # Take positive matches
        positive_sim = calculate_similarity_vector(kp1_desc, w_kp1_desc)
        positive_sim = positive_sim.view(-1, 1).repeat(1, 4).view(-1)

        # Create neighbour mask
        kp_grid = kp2coord(w_kp1).unsqueeze(1)
        coo_grid = create_coordinates_grid(desc2.size()).view(-1, 2).unsqueeze(0).to(desc2.device)
        coo_grid = coo_grid * self.grid_size + self.grid_size // 2

        grid_dist = torch.norm(kp_grid - coo_grid, dim=-1)
        _, ids = grid_dist.topk(k=4, largest=False, dim=-1)

        r_ids = torch.arange(0, ids.size(0)).view(-1, 1).repeat(1, ids.size(1)).view(-1)

        mask = torch.zeros_like(grid_dist)
        mask[r_ids, ids.view(-1)] = 1.0

        # Calculate similarity
        desc2 = desc2.permute((0, 2, 3, 1)).view(-1, desc2.size(1))
        desc_sim = calculate_similarity_matrix(kp1_desc, desc2)

        # Apply neighbour mask and get negatives
        desc_sim = desc_sim + mask * 5
        # neg_sim = desc_sim.min(dim=-1)[0]
        neg_sim = desc_sim.topk(k=4, dim=-1, largest=False)[0].view(-1)

        loss = torch.clamp(positive_sim - neg_sim + self.margin, min=0).mean() * self.loss_lambda

        return loss, positive_sim.mean()
