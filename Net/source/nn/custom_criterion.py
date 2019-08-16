import torch
import torch.nn as nn

from Net.source.utils.image_utils import (create_desc_coordinates_grid,
                                          warp_points)

from Net.source.utils.model_utils import sample_descriptors
from Net.source.utils.math_utils import calculate_inv_similarity_matrix, calculate_inv_similarity_vector, \
    calculate_distance_matrix


class HardTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, num_neg, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin
        self.num_neg = num_neg

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
        b, n, c = kp1_desc.size()

        w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)

        # Take positive matches
        positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
        positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, self.num_neg).view(b, n * self.num_neg)

        # Create neighbour mask
        coo_grid = create_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)

        d1_inter_num = 4

        grid_dist1 = calculate_distance_matrix(kp1, coo_grid)
        _, kp1_cell_ids = grid_dist1.topk(k=d1_inter_num, largest=False, dim=-1)

        kp1_cell_ids = kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
        kp1_cells = coo_grid.gather(dim=1, index=kp1_cell_ids)

        w_kp1_cells = warp_points(kp1_cells, homo12)

        d2_inter_num = 4

        grid_dist2 = calculate_distance_matrix(w_kp1_cells, coo_grid)
        _, kp2_cell_ids = grid_dist2.topk(k=d2_inter_num, largest=False, dim=-1)

        neigh_mask = torch.zeros_like(grid_dist2).to(grid_dist2)
        neigh_mask = neigh_mask.scatter(dim=-1, index=kp2_cell_ids, value=1)
        neigh_mask = neigh_mask.view(b, n, d2_inter_num, -1).sum(dim=2).float()

        # Calculate similarity
        desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
        desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)

        #  Apply neighbour mask and get negatives
        desc_sim = desc_sim + neigh_mask * 5
        neg_sim = desc_sim.topk(k=self.num_neg, dim=-1, largest=False)[0].view(b, -1)

        loss = torch.clamp(positive_sim - neg_sim + self.margin, min=0).mean() * self.loss_lambda

        return loss


class HardQuadTripletLoss(HardTripletLoss):

    def __init__(self, grid_size, margin, num_neg, loss_lambda):
        super().__init__(grid_size, margin, num_neg, loss_lambda)

    def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
        """
        :param kp1 B x N x 2
        :param w_kp1: B x N x 2
        :param kp1_desc: B x N x C
        :param desc2: B x C x H x W
        :param homo12: B x 3 x 3
        :return: float
        """
        b, n, c = kp1_desc.size()

        w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)

        # Take positive matches
        positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
        positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, self.num_neg).view(b, n * self.num_neg)

        # Create neighbour mask
        coo_grid = create_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)

        d1_inter_num = 4

        grid_dist1 = calculate_distance_matrix(kp1, coo_grid)
        _, kp1_cell_ids = grid_dist1.topk(k=d1_inter_num, largest=False, dim=-1)

        kp1_cell_ids = kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
        kp1_cells = coo_grid.gather(dim=1, index=kp1_cell_ids)

        w_kp1_cells = warp_points(kp1_cells, homo12)

        d2_inter_num = 4

        grid_dist2 = calculate_distance_matrix(w_kp1_cells, coo_grid)
        _, kp2_cell_ids = grid_dist2.topk(k=d2_inter_num, largest=False, dim=-1)

        neigh_mask = torch.zeros_like(grid_dist2).to(grid_dist2)
        neigh_mask = neigh_mask.scatter(dim=-1, index=kp2_cell_ids, value=1)
        neigh_mask = neigh_mask.view(b, n, d2_inter_num, -1).sum(dim=2).float()

        # Calculate similarity
        desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
        desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)

        #  Apply neighbour mask and get negatives
        desc_sim = desc_sim + neigh_mask * 5
        neg_sim = desc_sim.topk(k=self.num_neg, dim=-1, largest=False)[0].view(b, -1)

        loss = (torch.clamp(positive_sim - neg_sim + self.margin, min=0)**2).mean() * self.loss_lambda

        return loss