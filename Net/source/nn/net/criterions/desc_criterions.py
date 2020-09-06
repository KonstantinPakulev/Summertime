import torch
from torch import nn as nn

import Net.source.core.experiment as exp

from Net.source.utils.math_utils import smooth_inv_cos_sim_mat, smooth_inv_cos_sim_vec
from Net.source.nn.net.utils.endpoint_utils import sample_descriptors
from Net.source.nn.net.utils.criterion_utils import create_neigh_mask_ids


# TODO. Try L2-norm
# TODO. Keypoints based descriptor learning?

class DescTripletLoss(nn.Module):

    @staticmethod
    def from_config(model_config, criterion_config):
        desc_config = criterion_config[exp.DESC]
        return DescTripletLoss(model_config[exp.GRID_SIZE], desc_config[exp.MARGIN], desc_config[exp.LAMBDA])

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin
        self.loss_lambda = loss_lambda

    def forward(self, desc1, desc2, w_desc_grid1, w_vis_desc_grid_mask1):
        b, c, hc, wc = desc1.size()

        w_desc1 = sample_descriptors(desc2, w_desc_grid1, self.grid_size)

        neigh_mask_ids = create_neigh_mask_ids(w_desc_grid1, desc2.shape, self.grid_size)

        flat = hc * wc
        neigh_mask = torch.zeros(b, flat, flat).to(neigh_mask_ids.device)
        neigh_mask = neigh_mask.scatter(dim=-1, index=neigh_mask_ids, value=1)

        # Get positive and negative matches
        desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(b, c, -1).permute((0, 2, 1))

        desc_sim = smooth_inv_cos_sim_mat(desc1, desc2)
        desc_sim = desc_sim + neigh_mask * 5

        pos_sim = smooth_inv_cos_sim_vec(desc1, w_desc1)
        neg_sim = desc_sim.min(dim=-1)[0]

        loss = torch.clamp(pos_sim - neg_sim + self.margin, min=0) * w_vis_desc_grid_mask1.float()  # B x N
        loss = loss.sum(dim=-1) / w_vis_desc_grid_mask1.float().sum(dim=-1).clamp(min=1e-8)  # B
        loss = self.loss_lambda * loss.mean()

        return loss, pos_sim, neg_sim
