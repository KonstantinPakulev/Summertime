import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.source.utils.image_utils import (create_desc_coordinates_grid,
                                          warp_points,
                                          warp_image,
                                          gaussian_filter,
                                          select_keypoints)

from Net.source.utils.model_utils import sample_descriptors
from Net.source.utils.math_utils import calculate_inv_similarity_matrix, calculate_inv_similarity_vector, \
    calculate_distance_matrix


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

        _, kp1 = select_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)

        gt_score1, _ = select_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
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
        b, n, c = kp1_desc.size()

        w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)

        # Take positive matches
        positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
        positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, self.num_neg).view(b, n * self.num_neg)

        # Create neighbour mask
        coo_grid = create_desc_coordinates_grid(desc2, self.grid_size).view(desc2.size(0), -1, 2).to(desc2.device)

        kp1_coo_dist = calculate_distance_matrix(kp1, coo_grid)
        _, kp1_cell_ids = kp1_coo_dist.topk(k=4, largest=False, dim=-1)

        kp1_cell_ids = kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
        kp1_cells = coo_grid.gather(dim=1, index=kp1_cell_ids)

        kp1_w_cells = warp_points(kp1_cells, homo12)

        kp1_wc_coo_dist = calculate_distance_matrix(kp1_w_cells, coo_grid)
        _, kp1_w_cell_cell_ids = kp1_wc_coo_dist.topk(k=4, largest=False, dim=-1)

        neigh_mask = torch.zeros_like(kp1_wc_coo_dist).to(kp1_wc_coo_dist)
        neigh_mask = neigh_mask.scatter(dim=-1, index=kp1_w_cell_cell_ids, value=1)
        neigh_mask = neigh_mask.view(b, n, 4, -1).sum(dim=2).float()

        # Calculate similarity
        desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
        desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)

        #  Apply neighbour mask and get negatives
        desc_sim = desc_sim + neigh_mask * 5
        neg_sim = desc_sim.topk(k=self.num_neg, dim=-1, largest=False)[0].view(b, -1)

        fos = (torch.clamp(positive_sim - neg_sim + self.margin, min=0) ** 2).mean() * self.loss_lambda

        # Calculate SOS
        # Create kp1 neighbour mask
        kp1_cells = kp1_cells.view(b, n * 4, 2)
        kp1_cells_dist = calculate_distance_matrix(kp1_cells, kp1_cells)
        kp1_mask = (kp1_cells_dist <= 1e-8).view(b, n, 4, n, 4).sum(-1).sum(2).float()

        # Create w_kp1 neighbour mask
        kp1_w_cell_cell_ids = kp1_w_cell_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
        kp1_w_cell_cells = coo_grid.gather(dim=1, index=kp1_w_cell_cell_ids)

        w_kp1_coo_dist = calculate_distance_matrix(w_kp1, coo_grid)
        _, w_kp1_cell_ids = w_kp1_coo_dist.topk(k=4, largest=False, dim=-1)

        w_kp1_cell_ids = w_kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
        w_kp1_cells = coo_grid.gather(dim=1, index=w_kp1_cell_ids)

        kp1_wc_w_kp1_dist = calculate_distance_matrix(kp1_w_cell_cells, w_kp1_cells)
        w_kp1_mask = (kp1_wc_w_kp1_dist <= 1e-8).view(b, n, 4, 4, n, 4).sum(-1).sum(3).sum(2).float()

        kp1_sim = calculate_inv_similarity_matrix(kp1_desc, kp1_desc)
        w_kp1_sim = calculate_inv_similarity_matrix(w_kp1_desc, w_kp1_desc)

        kp1_sim = kp1_sim + kp1_mask * 5
        w_kp1_sim = w_kp1_sim + w_kp1_mask * 5

        _, kp1_neg_ids = kp1_sim.topk(k=self.sos_neg, dim=-1, largest=False)
        _, w_kp1_neg_ids = w_kp1_sim.topk(k=self.sos_neg, dim=-1, largest=False)

        kp1_neg_ids = kp1_neg_ids.view(b, n * self.sos_neg).unsqueeze(-1).repeat(1, 1, c)
        w_kp1_neg_ids = w_kp1_neg_ids.view(b, n * self.sos_neg).unsqueeze(-1).repeat(1, 1, c)

        kp1_neg_desc = kp1_desc.gather(dim=1, index=kp1_neg_ids)
        w_kp1_neg_desc = w_kp1_desc.gather(dim=1, index=w_kp1_neg_ids)

        kp1_desc = kp1_desc.unsqueeze(2).repeat(1, 1, self.sos_neg, 1).view(b, n * self.sos_neg, c)
        w_kp1_desc = w_kp1_desc.unsqueeze(2).repeat(1, 1, self.sos_neg, 1).view(b, n * self.sos_neg, c)

        sos = calculate_inv_similarity_vector(kp1_desc, kp1_neg_desc) - \
              calculate_inv_similarity_vector(w_kp1_desc, w_kp1_neg_desc)

        sos = (sos ** 2).view(b, n, self.sos_neg).sum(-1).sqrt().mean()

        loss = (fos + sos) * self.loss_lambda

        return loss


class SimpleHardQuadTripletSOSRLoss(nn.Module):

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
        b, n, c = kp1_desc.size()

        w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)

        # Take positive matches
        positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
        positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, self.num_neg).view(b, n * self.num_neg)

        # Create neighbour mask
        coo_grid = create_desc_coordinates_grid(desc2, self.grid_size).view(desc2.size(0), -1, 2).to(desc2.device)

        kp1_coo_dist = calculate_distance_matrix(kp1, coo_grid)
        _, kp1_cell_ids = kp1_coo_dist.topk(k=4, largest=False, dim=-1)

        kp1_cell_ids = kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
        kp1_cells = coo_grid.gather(dim=1, index=kp1_cell_ids)

        kp1_w_cells = warp_points(kp1_cells, homo12)

        kp1_wc_coo_dist = calculate_distance_matrix(kp1_w_cells, coo_grid)
        _, kp1_w_cell_cell_ids = kp1_wc_coo_dist.topk(k=4, largest=False, dim=-1)

        neigh_mask = torch.zeros_like(kp1_wc_coo_dist).to(kp1_wc_coo_dist)
        neigh_mask = neigh_mask.scatter(dim=-1, index=kp1_w_cell_cell_ids, value=1)
        neigh_mask = neigh_mask.view(b, n, 4, -1).sum(dim=2).float()

        # Calculate similarity
        desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
        desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)

        #  Apply neighbour mask and get negatives
        desc_sim = desc_sim + neigh_mask * 5
        neg_sim = desc_sim.topk(k=self.num_neg, dim=-1, largest=False)[0].view(b, -1)

        fos = (torch.clamp(positive_sim - neg_sim + self.margin, min=0) ** 2).mean() * self.loss_lambda

        radius = self.grid_size * np.sqrt(2) + 0.1

        kp1_sim = calculate_inv_similarity_matrix(kp1_desc, kp1_desc)
        kp1_mask = (calculate_distance_matrix(kp1, kp1) <= radius).float()

        w_kp1_sim = calculate_inv_similarity_matrix(w_kp1_desc, w_kp1_desc)
        w_kp1_mask = (calculate_distance_matrix(w_kp1, w_kp1) <= radius).float()

        kp1_sim = kp1_sim + kp1_mask * 5
        w_kp1_sim = w_kp1_sim + w_kp1_mask * 5

        _, kp1_neg_ids = kp1_sim.topk(k=self.sos_neg, dim=-1, largest=False)
        _, w_kp1_neg_ids = w_kp1_sim.topk(k=self.sos_neg, dim=-1, largest=False)

        kp1_neg_ids = kp1_neg_ids.view(b, n * self.sos_neg).unsqueeze(-1).repeat(1, 1, c)
        w_kp1_neg_ids = w_kp1_neg_ids.view(b, n * self.sos_neg).unsqueeze(-1).repeat(1, 1, c)

        kp1_neg_desc = kp1_desc.gather(dim=1, index=kp1_neg_ids)
        w_kp1_neg_desc = w_kp1_desc.gather(dim=1, index=w_kp1_neg_ids)

        kp1_desc = kp1_desc.unsqueeze(2).repeat(1, 1, self.sos_neg, 1).view(b, n * self.sos_neg, c)
        w_kp1_desc = w_kp1_desc.unsqueeze(2).repeat(1, 1, self.sos_neg, 1).view(b, n * self.sos_neg, c)

        sos = calculate_inv_similarity_vector(kp1_desc, kp1_neg_desc) - calculate_inv_similarity_vector(w_kp1_desc,
                                                                                                        w_kp1_neg_desc)

        sos = (sos ** 2).view(b, n, self.sos_neg).sum(-1).sqrt().mean()

        loss = (fos + sos) * self.loss_lambda

        return loss
