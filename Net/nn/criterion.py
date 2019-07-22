import torch
import torch.nn as nn

from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid


class HomoHingeLoss(nn.Module):

    def __init__(self, grid_size, pos_lambda, pos_margin, neg_margin):
        super().__init__()
        self.grid_size = grid_size
        self.pos_lambda = pos_lambda
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, desc1, desc2, homo):
        """
        :param desc1: N x C x H/8 x W/8
        :param desc2: N x C x H/8 x W/8
        :param homo: 3 x 3
        """

        # We need to account for difference in size between descriptor and image for homography to work
        grid = create_coordinates_grid(desc1.size()) * self.grid_size + self.grid_size // 2
        w_grid = warp_coordinates_grid(grid, homo)

        grid = grid.unsqueeze(dim=3).unsqueeze(dim=3)
        w_grid = w_grid.unsqueeze(dim=1).unsqueeze(dim=1)

        grid_dist = torch.norm(grid - w_grid, dim=-1)
        ones = torch.ones_like(grid_dist)
        zeros = torch.zeros_like(grid_dist)
        # Mask with homography induced correspondences
        s = torch.where(grid_dist <= self.grid_size - 0.5, ones, zeros)

        desc1 = desc1.unsqueeze(4).unsqueeze(4)
        desc2 = desc2.unsqueeze(2).unsqueeze(2)
        dot_desc = torch.sum(desc1 * desc2, dim=1)

        pos_dist = (self.pos_margin - dot_desc).clamp(min=0)
        neg_dist = (dot_desc - self.neg_margin).clamp(min=0)

        loss = self.pos_lambda * s * pos_dist + (1 - s) * neg_dist

        # TODO. Bordering mask

        return loss.mean()