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
                                   erode_filter,
                                   gaussian_filter,
                                   filter_border,
                                   select_keypoints)

from Net.utils.model_utils import sample_descriptors, space_to_depth
from Net.utils.math_utils import calculate_similarity_matrix, calculate_similarity_vector, calculate_distance_matrix


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

        # Create visibility mask of the first image
        vis_mask = erode_filter(torch.ones_like(score1).to(score1.device))
        w_vis_mask = warp_image(score1, torch.ones_like(score2).to(score2.device), homo12).gt(0).float()
        w_vis_mask = erode_filter(w_vis_mask)

        # Filter borders
        score1 = score1 * vis_mask
        w_score2 = w_score2 * w_vis_mask

        # Extract keypoints and prepare ground truth
        score1, kp1 = select_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)
        score1 = gaussian_filter(score1, self.gauss_k_size, self.gauss_sigma)

        gt_score1, _ = select_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        norm = w_vis_mask.sum()
        loss = F.mse_loss(score1, gt_score1, reduction='none') * w_vis_mask / norm
        loss = loss.sum() * self.loss_lambda

        return loss, kp1


class HardTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin
        # TODO. Make it a parameter
        self.num_neg = 4

        self.loss_lambda = loss_lambda

    def forward(self, w_kp1, kp1_desc, desc2):
        """
        :param w_kp1: B x N x 2
        :param kp1_desc: B x N x C
        :param desc2: B x C x H x W
        :return: float
        """
        b, n, c = kp1_desc.size()

        # Calculate similarity measure between anchors and positives
        w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)

        # Take positive matches
        positive_sim = calculate_similarity_vector(kp1_desc, w_kp1_desc)
        positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, self.num_neg).view(b, n * self.num_neg)

        # Create neighbour mask
        kp_grid = w_kp1[:, :, [1, 0]]
        coo_grid = create_coordinates_grid(desc2).view(b, -1, 2).to(desc2.device)
        coo_grid = coo_grid * self.grid_size + self.grid_size // 2

        grid_dist = calculate_distance_matrix(kp_grid, coo_grid)
        _, ids = grid_dist.topk(k=4, largest=False, dim=-1)

        mask = torch.zeros_like(grid_dist).to(grid_dist.device)
        mask = mask.scatter(dim=-1, index=ids, value=1)

        # Calculate similarity
        desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
        desc_sim = calculate_similarity_matrix(kp1_desc, desc2)

        # Apply neighbour mask and get negatives
        desc_sim = desc_sim + mask * 5

        neg_sim = desc_sim.topk(k=4, dim=-1, largest=False)[0].view(b, -1)

        loss = torch.clamp(positive_sim - neg_sim + self.margin, min=0).mean() * self.loss_lambda

        return loss
