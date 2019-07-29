import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.utils.model_utils import space_to_depth
from Net.utils.image_utils import (create_coordinates_grid,
                                   warp_coordinates_grid,
                                   warp_image,
                                   erode_filter,
                                   dilate_filter,
                                   gaussian_filter,
                                   select_keypoints)


class HomoMSELoss(nn.Module):

    def __init__(self, nms_thresh, nms_k_size, top_k, gauss_k_size, gauss_sigma):
        super().__init__()

        self.nms_thresh = nms_thresh
        self.nms_k_size = nms_k_size

        self.top_k = top_k

        self.gauss_k_size = gauss_k_size
        self.gauss_sigma = gauss_sigma

    def forward(self, score1, score2, homo):

        # Warp score2 to score1 space
        w_score2 = warp_image(score2, homo)

        # Create visibility mask of first image and remove bordering artifacts
        vis_mask1 = warp_image(torch.ones_like(score2).to(homo.device), homo).gt(0).float()
        vis_mask1 = erode_filter(vis_mask1)

        # Apply visibility mask
        score1 = score1 * vis_mask1
        w_score2 = w_score2 * vis_mask1

        # Extract keypoints and prepare ground truth
        _, kp1 = select_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1, _ = select_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        norm = vis_mask1.sum()
        loss = F.mse_loss(score1, gt_score1, reduction='none') / norm
        loss = loss.sum()

        return loss, kp1, vis_mask1


class HomoHingeLoss(nn.Module):

    def __init__(self, grid_size, pos_lambda, pos_margin, neg_margin):
        super().__init__()
        self.grid_size = grid_size

        self.pos_lambda = pos_lambda

        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, desc1, desc2, homo, vis_mask1):
        """
        :param desc1: N x C x Hr x Wr
        :param desc2: N x C x Hr x Wr
        :param homo: 3 x 3
        :param vis_mask1: Mask of the first image. N x 1 x H x W
        Note: 'r' suffix means reduced in 'grid_size' times
        """

        # We need to account for difference in size between descriptor and image for homography to work
        grid = create_coordinates_grid(desc1.size()) * self.grid_size + self.grid_size // 2
        grid = grid.type_as(homo).to(homo.device)
        w_grid = warp_coordinates_grid(grid, homo)

        grid = grid.unsqueeze(dim=3).unsqueeze(dim=3)
        w_grid = w_grid.unsqueeze(dim=1).unsqueeze(dim=1)

        n, _, hr, wr = desc1.size()

        # Reduce spatial dimensions of visibility mask
        vis_mask1 = space_to_depth(vis_mask1, self.grid_size).prod(dim=1)
        vis_mask1 = vis_mask1.reshape([n, 1, 1, hr, wr])

        # Mask with homography induced correspondences
        grid_dist = torch.norm(grid - w_grid, dim=-1)
        ones = torch.ones_like(grid_dist)
        zeros = torch.zeros_like(grid_dist)
        s = torch.where(grid_dist <= self.grid_size - 0.5, ones, zeros)

        # Create negative correspondences around positives
        ns = s.clone().view(-1, 1, hr, wr)
        ns = dilate_filter(ns).view(n, hr, wr, hr, wr)
        ns = ns - s
        # ns = 1 - s

        # Apply visibility mask
        s *= vis_mask1
        ns *= vis_mask1

        desc1 = desc1.unsqueeze(4).unsqueeze(4)
        desc2 = desc2.unsqueeze(2).unsqueeze(2)
        dot_desc = torch.sum(desc1 * desc2, dim=1)

        pos_dist = (self.pos_margin - dot_desc).clamp(min=0)
        neg_dist = (dot_desc - self.neg_margin).clamp(min=0)

        loss = self.pos_lambda * s * pos_dist + ns * neg_dist

        norm = hr * wr * 4
        # norm = hr * wr * r_mask.sum()
        loss = loss.sum() / norm

        return loss
