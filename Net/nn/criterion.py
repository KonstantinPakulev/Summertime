import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.utils.image_utils import (create_coordinates_grid,
                                   warp_coordinates_grid,
                                   warp_image,
                                   erode_filter,
                                   dilate_filter,
                                   gaussian_filter,
                                   nms,
                                   space_to_depth)


class HomoMSELoss(nn.Module):

    def __init__(self, nms_thresh, nms_k_size, top_k, gauss_k_size, gauss_sigma):
        super().__init__()

        self.nms_thresh = nms_thresh
        self.nms_k_size = nms_k_size

        self.top_k = top_k

        self.gauss_k_size = gauss_k_size
        self.gauss_sigma = gauss_sigma

    def prepare_score_map(self, det):
        n, c, h, w = det.size()
        flat = h * w

        # Determine top k activations in w_det2
        det = nms(det, self.nms_thresh, self.nms_k_size).view(n, c, flat)
        _, top_k_indices = torch.topk(det, self.top_k)

        # Create top k activations mask
        top_k_mask = torch.zeros_like(det).to(det.device)
        top_k_mask[:, :, top_k_indices] = 1.0

        # Select top k activations from w_det2
        det = det * top_k_mask
        det = det.view(n, c, h, w)
        top_k_mask = top_k_mask.view(n, c, h, w)

        det = gaussian_filter(det, self.gauss_k_size, self.gauss_sigma)

        return det, top_k_mask

    def forward(self, det1, det2, homo):
        # Warp detections to create ground truth for det1
        w_det2 = warp_image(det2, homo)

        # Create visibility mask and remove bordering artifacts
        vis_mask = warp_image(torch.ones_like(det2).to(homo.device), homo).gt(0).float()
        vis_mask = erode_filter(vis_mask)

        det1, _ = self.prepare_score_map(det1)
        w_det2, top_k_mask = self.prepare_score_map(w_det2)

        norm = vis_mask.sum()
        loss = F.mse_loss(det1, w_det2, reduction='none') * vis_mask / norm
        loss = loss.sum()

        return loss, top_k_mask, vis_mask


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

        n, _, hr, wr = des2.size()

        # Reduce spatial dimensions of visibility mask
        r_mask = space_to_depth(mask, self.grid_size).prod(dim=1)
        r_mask = r_mask.reshape([n, 1, 1, hr, wr])

        # Mask with homography induced correspondences
        grid_dist = torch.norm(grid - w_grid, dim=-1)
        ones = torch.ones_like(grid_dist)
        zeros = torch.zeros_like(grid_dist)
        s = torch.where(grid_dist <= self.grid_size - 0.5, ones, zeros)

        # Apply visibility mask
        s *= r_mask

        # Create negative correspondences around positives
        ns = s.clone().view(-1, 1, hr, wr)
        ns = dilate_filter(ns).view(n, hr, wr, hr, wr)
        ns = ns - s
        # ns = 1 - s

        # Apply visibility mask
        ns *= r_mask

        des1 = des1.unsqueeze(4).unsqueeze(4)
        des2 = des2.unsqueeze(2).unsqueeze(2)
        dot_des = torch.sum(des1 * des2, dim=1)

        pos_dist = (self.pos_margin - dot_des).clamp(min=0)
        neg_dist = (dot_des - self.neg_margin).clamp(min=0)

        loss = self.pos_lambda * s * pos_dist + ns * neg_dist

        norm = hr * wr * 4
        # norm = hr * wr * r_mask.sum()
        loss = loss.sum() / norm

        return loss, s, dot_des
