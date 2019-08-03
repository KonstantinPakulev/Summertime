import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.utils.image_utils import (warp_keypoints,
                                   warp_image,
                                   gaussian_filter,
                                   filter_border,
                                   select_keypoints)
from Net.utils.model_utils import sample_descriptors


class HomoMSELoss(nn.Module):

    def __init__(self, nms_thresh, nms_k_size, top_k, gauss_k_size, gauss_sigma):
        super().__init__()

        self.nms_thresh = nms_thresh
        self.nms_k_size = nms_k_size

        self.top_k = top_k

        self.gauss_k_size = gauss_k_size
        self.gauss_sigma = gauss_sigma

    def forward(self, score1, score2, homo12):
        # Filter border
        score2 = filter_border(score2)

        # Warp score2 to score1 space
        w_score2 = warp_image(score1, score2, homo12)

        # Create visibility mask of the first image
        vis_mask = torch.ones_like(score2).to(score2.device)
        vis_mask1 = warp_image(score1, vis_mask, homo12).gt(0).float()

        # Extract keypoints and prepare ground truth
        score1, kp1 = select_keypoints(score1, self.nms_thresh, self.nms_k_size, self.top_k)
        score1 = gaussian_filter(score1, self.gauss_k_size, self.gauss_sigma)

        gt_score1, _ = select_keypoints(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
        gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)

        norm = vis_mask1.sum()
        loss = F.mse_loss(score1, gt_score1, reduction='none') * vis_mask1 / norm
        loss = loss.sum()

        return loss, kp1


class HomoTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, neighbour_thresh):
        super().__init__()

        self.grid_size = grid_size
        self.margin = margin
        self.neighbour_thresh = neighbour_thresh

    def forward(self, kp1, desc1, desc2, homo12):
        # Sample anchor descriptors
        anchor_desc = sample_descriptors(desc1, kp1, self.grid_size)

        # Sample positive descriptors
        w_kp1 = warp_keypoints(kp1, homo12)
        positive_desc = sample_descriptors(desc2, w_kp1, self.grid_size)

        dot_desc = torch.mm(anchor_desc, positive_desc.t())
        positive = dot_desc.diag()

        factor = 3

        # Remove diagonal from calculations
        dot_desc -= torch.eye(anchor_desc.size(0)).to(anchor_desc.device) * factor

        # Also remove elements that are to close anchor and positive keypoints to not to pick them as negatives
        anchor_n_mask = torch.pairwise_distance(kp1[:, 2:].float(), kp1[:, 2:].float()).lt(self.neighbour_thresh)
        dot_desc -= anchor_n_mask.float() * factor

        positive_n_mask = torch.pairwise_distance(w_kp1[:, 2:].float(), w_kp1[:, 2:].float()).lt(self.neighbour_thresh)
        dot_desc -= positive_n_mask.float() * factor

        rn = dot_desc.max(0)[0]
        cn = dot_desc.max(1)[0]
        negative = torch.max(rn, cn)

        triplet_loss = torch.clamp(positive.sum() - negative + self.margin, min=0.0).mean()

        return triplet_loss, anchor_desc, w_kp1


class HomoHingeLoss(nn.Module):

    def __init__(self, grid_size, pos_lambda, pos_margin, neg_margin):
        super().__init__()
        self.grid_size = grid_size

        self.pos_lambda = pos_lambda

        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, kp1, desc1, desc2, homo12):
        """
        :param desc1: N x C x Hr x Wr
        :param desc2: N x C x Hr x Wr
        :param homo21: N x 3 x 3
        :param vis_mask1: Mask of the first image. N x 1 x H x W
        Note: 'r' suffix means reduced in 'grid_size' times
        """
        w_kp1 = warp_keypoints(kp1, homo12)
        positive_desc = sample_descriptors(desc2, w_kp1, self.grid_size)


        pos_dist = (self.pos_margin - dot_desc).clamp(min=0)
        neg_dist = (dot_desc - self.neg_margin).clamp(min=0)

        loss = self.pos_lambda * s * pos_dist + ns * neg_dist

        norm = hr * wr * vis_mask1.sum()
        loss = loss.sum() / norm

        return loss










