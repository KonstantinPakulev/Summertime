import torch
import torch.nn as nn

from Net.utils.image_utils import (create_coordinates_grid,
                                   warp_coordinates_grid,
                                   dilate_mask,
                                   space_to_depth)


class HomoHingeLoss(nn.Module):

    def __init__(self, grid_size, pos_lambda, pos_margin, neg_margin):
        super().__init__()
        self.grid_size = grid_size
        self.pos_lambda = pos_lambda
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, des1, des2, homo, mask):
        """
        :param des1: N x C x Hr x Wr
        :param des2: N x C x Hr x Wr
        :param homo: 3 x 3
        :param mask: Mask of the second image. N x 1 x H x W
        Note: 'r' suffix means reduced in 'grid_size' times
        """

        # We need to account for difference in size between descriptor and image for homography to work
        grid = create_coordinates_grid(des2.size()) * self.grid_size + self.grid_size // 2
        grid = grid.type_as(homo).to(homo.device)
        w_grid = warp_coordinates_grid(grid, homo)

        grid = grid.unsqueeze(dim=3).unsqueeze(dim=3)
        w_grid = w_grid.unsqueeze(dim=1).unsqueeze(dim=1)

        # Mask with homography induced correspondences
        grid_dist = torch.norm(grid - w_grid, dim=-1)
        ones = torch.ones_like(grid_dist)
        zeros = torch.zeros_like(grid_dist)
        s = torch.where(grid_dist <= self.grid_size - 0.5, ones, zeros)

        n, _, hr, wr = des2.size()

        # Create negative correspondences around positives
        # ns = s.clone().view(-1, 1, hr, wr)
        # ns = dilate_mask(ns).view(n, hr, wr, hr, wr)
        # ns = ns - s

        des1 = des1.unsqueeze(4).unsqueeze(4)
        des2 = des2.unsqueeze(2).unsqueeze(2)
        dot_des = torch.sum(des1 * des2, dim=1)

        pos_dist = (self.pos_margin - dot_des).clamp(min=0)
        neg_dist = (dot_des - self.neg_margin).clamp(min=0)

        # loss = self.pos_lambda * s * pos_dist + ns * neg_dist
        loss = self.pos_lambda * s * pos_dist + (1 - s) * neg_dist

        # Mask bordering pixels
        r_mask = space_to_depth(mask, self.grid_size).prod(dim=1)
        r_mask = r_mask.reshape([n, 1, 1, hr, wr])

        # norm = hr * wr * 4
        norm = hr * wr * r_mask.sum()
        loss = torch.sum(loss * r_mask) / norm

        return loss, s, dot_des, r_mask.squeeze(1)
