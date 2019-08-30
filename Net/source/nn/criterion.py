import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.source.utils.model_utils import sample_descriptors, space_to_depth
from Net.source.utils.image_utils import (warp_image,
                                          create_desc_coordinates_grid,
                                          create_center_desc_coordinates_grid,
                                          warp_coordinates_grid,
                                          gaussian_filter,
                                          prepare_gt_score,
                                          get_visible_keypoints_mask)
from Net.source.utils.math_utils import calculate_distance_matrix, calculate_inv_similarity_matrix, \
    calculate_inv_similarity_vector


class MSELoss(nn.Module):

    def __init__(self, nms_thresh, nms_k_size, top_k, gauss_k_size, gauss_sigma, loss_lambda):
        super().__init__()

        self.nms_thresh = nms_thresh
        self.nms_k_size = nms_k_size

        self.top_k = top_k

        self.gauss_k_size = gauss_k_size
        self.gauss_sigma = gauss_sigma

        self.loss_lambda = loss_lambda

    def forward(self, score1, score2, w_vis_mask2, homo12):
        """
        :param score1: B x 1 x H x W
        :param score2: B x 1 x H x W
        :param homo12: B x 3 x 3
        :param w_vis_mask2: B x 1 x H x W
        :return: float, B x C x N x 2
        """
        # Warp score2 to score1 space
        w_score2 = warp_image(score1, score2, homo12)

        gt_score1 = prepare_gt_score(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        loss = F.mse_loss(score1, gt_score1, reduction='none') * w_vis_mask2.float()
        loss = loss.sum() * self.loss_lambda / w_vis_mask2.sum()

        return loss


class DenseInterQTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin

        self.loss_lambda = loss_lambda


    def forward(self, desc1, desc2, homo12, w_vis_mask1, score2):
        b, c, hc, wc = desc1.size()
        flat = hc * wc

        coo_grid1 = create_desc_coordinates_grid(desc1, self.grid_size, False).to(desc1.device)
        w_coo_grid1 = warp_coordinates_grid(coo_grid1, homo12).view(b, -1, 2)
        w_coo_grid1 = w_coo_grid1[:, :, [1, 0]]

        w_desc1 = sample_descriptors(desc2, w_coo_grid1, self.grid_size)

        # Create neigh mask
        coo_grid2 = create_center_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
        coo_dist = calculate_distance_matrix(w_coo_grid1, coo_grid2)
        _, ul = coo_dist.min(dim=-1)

        ul = ul.unsqueeze(-1)

        ur = ul + 1
        ur = torch.where(ur >= flat, ul, ur)

        ll = ul + wc
        ll = torch.where(ll >= flat, ul, ll)

        lr = ll + 1
        lr = torch.where(lr >= flat, ul, lr)

        neigh_mask_ids = torch.cat([ul, ur, ll, lr], dim=-1)

        neigh_mask = torch.zeros_like(coo_dist).to(coo_dist.device)
        neigh_mask = neigh_mask.scatter(dim=-1, index=neigh_mask_ids, value=1)

        # Prepare visibility mask
        w_vis_mask1 = space_to_depth(w_vis_mask1, self.grid_size)
        w_vis_mask1 = w_vis_mask1.prod(dim=1).unsqueeze(1).view(b, 1, -1)

        # Get positive and negative matches
        desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(b, c, -1).permute((0, 2, 1))

        desc_sim = calculate_inv_similarity_matrix(desc1, desc2)
        desc_sim = desc_sim + neigh_mask * 5
        desc_sim = desc_sim + (1 - w_vis_mask1.float()) * 5

        wv_match_mask1 = get_visible_keypoints_mask(score2, w_coo_grid1).float()

        pos_sim = calculate_inv_similarity_vector(desc1, w_desc1)
        neg_sim = desc_sim.min(dim=-1)[0]

        loss = torch.clamp(pos_sim - neg_sim + self.margin, min=0) ** 2 * wv_match_mask1
        loss = loss.sum() / wv_match_mask1.sum() * self.loss_lambda

        return loss