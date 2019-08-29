import torch
import torch.nn as nn

from Net.source.utils.model_utils import space_to_depth, sample_descriptors
from Net.source.utils.image_utils import create_desc_coordinates_grid, create_desc_center_coordinates_grid,\
    warp_coordinates_grid, get_visible_keypoints_mask, warp_image
from Net.source.utils.math_utils import calculate_distance_matrix, calculate_similarity_matrix, \
    calculate_inv_similarity_matrix, calculate_inv_similarity_vector


class PairHingeLoss(nn.Module):

    def __init__(self, grid_size, pos_lambda, pos_margin, neg_margin):
        super().__init__()

        self.grid_size = grid_size
        self.pos_lambda = pos_lambda
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, desc1, desc2, homo12):
        n, c, hc, wc = desc1.size()

        coo_grid1 = create_desc_center_coordinates_grid(desc1, self.grid_size, False).to(desc2.device)
        coo_grid2 = create_desc_center_coordinates_grid(desc2, self.grid_size, False).view(n, -1, 2).to(desc2.device)

        w_coo_grid1 = warp_coordinates_grid(coo_grid1, homo12).view(n, -1, 2)

        coo_dist = calculate_distance_matrix(w_coo_grid1, coo_grid2)
        pos_mask = (coo_dist <= self.grid_size).float()

        desc1 = desc1.view(n, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(n, c, -1).permute((0, 2, 1))

        coo_sim = calculate_similarity_matrix(desc1, desc2)

        pos_sim = (self.pos_margin - coo_sim).clamp(min=0.0) * pos_mask * self.pos_lambda
        neg_sim = (coo_sim - self.neg_margin).clamp(min=0.0) * (1 - pos_mask)

        loss = pos_sim + neg_sim

        norm = hc * hc * wc * wc
        loss = loss.sum() / norm

        return loss


class DenseTripletLoss(nn.Module):

    def __init__(self, grid_size, margin):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin

    def forward(self, score1, score2, desc1, desc2, homo12, homo21):
        b, c, hc, wc = desc1.size()

        coo_grid1 = create_desc_coordinates_grid(desc1, self.grid_size, False).to(desc1.device)
        w_coo_grid1 = warp_coordinates_grid(coo_grid1, homo12).view(b, -1, 2)
        w_coo_grid1 = w_coo_grid1[:, :, [1, 0]]

        w_desc1 = sample_descriptors(desc2, w_coo_grid1, self.grid_size)

        w_vis_mask = warp_image(score2, torch.ones_like(score1).to(score1.device), homo21).gt(0).float()
        w_vis_mask = space_to_depth(w_vis_mask, self.grid_size)
        w_vis_mask = w_vis_mask.prod(dim=1).unsqueeze(1).view(b, 1, -1)

        # Create neigh mask
        coo_grid2 = create_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
        coo_dist = calculate_distance_matrix(w_coo_grid1, coo_grid2)
        neigh_mask = (coo_dist <= self.grid_size).float()

        # Get positive and negative matches
        desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(b, c, -1).permute((0, 2, 1))

        desc_sim = calculate_inv_similarity_matrix(desc1, desc2)
        desc_sim = desc_sim + neigh_mask * 5
        desc_sim = desc_sim + (1 - w_vis_mask) * 5

        match_mask = get_visible_keypoints_mask(score2, w_coo_grid1).float()

        pos_sim = calculate_inv_similarity_vector(desc1, w_desc1)
        neg_sim = desc_sim.min(dim=-1)[0]

        loss = torch.clamp(pos_sim - neg_sim + self.margin, min=0) ** 2
        loss = (loss * match_mask).sum() / match_mask.sum()

        return loss


class DenseInterTripletLoss(nn.Module):

    def __init__(self, grid_size, margin):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin

    def forward(self, score1, score2, desc1, desc2, homo12, homo21):
        b, c, hc, wc = desc1.size()
        flat = hc * wc

        coo_grid1 = create_desc_coordinates_grid(desc1, self.grid_size, False).to(desc1.device)
        w_coo_grid1 = warp_coordinates_grid(coo_grid1, homo12).view(b, -1, 2)
        w_coo_grid1 = w_coo_grid1[:, :, [1, 0]]

        w_desc1 = sample_descriptors(desc2, w_coo_grid1, self.grid_size)

        w_vis_mask = warp_image(score2, torch.ones_like(score1).to(score1.device), homo21).gt(0).float()
        w_vis_mask = space_to_depth(w_vis_mask, self.grid_size)
        w_vis_mask = w_vis_mask.prod(dim=1).unsqueeze(1).view(b, 1, -1)

        # Create neigh mask
        coo_grid2 = create_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
        coo_dist = calculate_distance_matrix(w_coo_grid1, coo_grid2)
        _, w_cell_cells_ids = coo_dist.topk(k=4, largest=False, dim=-1)

        neigh_mask = torch.zeros_like(coo_dist).to(coo_dist.device)
        neigh_mask = neigh_mask.scatter(dim=-1, index=w_cell_cells_ids, value=1)

        # Get positive and negative matches
        desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(b, c, -1).permute((0, 2, 1))

        desc_sim = calculate_inv_similarity_matrix(desc1, desc2)
        desc_sim = desc_sim + neigh_mask * 5
        desc_sim = desc_sim + (1 - w_vis_mask) * 5

        match_mask = get_visible_keypoints_mask(score2, w_coo_grid1).float()

        pos_sim = calculate_inv_similarity_vector(desc1, w_desc1)
        neg_sim = desc_sim.min(dim=-1)[0]

        loss = torch.clamp(pos_sim - neg_sim + self.margin, min=0) ** 2
        loss = (loss * match_mask).sum() / match_mask.sum()

        return loss




