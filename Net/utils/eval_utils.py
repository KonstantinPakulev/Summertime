import os
import cv2
import numpy as np

import torch
from torchvision.utils import make_grid

"""
Mappings to process endpoint and calculate metrics
"""


def l_loss(x):
    return x['loss']


def l_det_loss(x):
    return x['det_loss']


def l_des_loss(x):
    return x['des_loss']


def l_collect_show(x):
    return x['im1'], x['im2'], x['top_k_mask1'], x['top_k_mask2']


# def l_nn_match(x):
#     ms1 = nearest_neighbor_match_score(x['s1'], x['dot_des1'], x['r_mask1'])
#     ms2 = nearest_neighbor_match_score(x['s2'], x['dot_des2'], x['r_mask2'])
#     return (ms1 + ms2) / 2


# def l_nn_match_2(x):
#     ms1 = nearest_neighbor_match_score(x['s1'], x['dot_des1'], x['r_mask1'], 2)
#     ms2 = nearest_neighbor_match_score(x['s2'], x['dot_des2'], x['r_mask2'], 2)
#     return (ms1 + ms2) / 2


# def l_nn_match_4(x):
#     ms1 = nearest_neighbor_match_score(x['s1'], x['dot_des1'], x['r_mask1'], 4)
#     ms2 = nearest_neighbor_match_score(x['s2'], x['dot_des2'], x['r_mask2'], 4)
#     return (ms1 + ms2) / 2


# def l_nnt_match(x):
#     ms1 = nearest_neighbor_thresh_match_score(x['des1'], x['des2'], cfg.METRIC.THRESH,
#                                               x['s1'], x['dot_des1'], x['r_mask1'])
#     ms2 = nearest_neighbor_thresh_match_score(x['des2'], x['des1'], cfg.METRIC.THRESH,
#                                               x['s2'], x['dot_des2'], x['r_mask2'])
#     return (ms1 + ms2) / 2

# def l_nnr_match(x):
#     ms1 = nearest_neighbor_ratio_match_score(x['des1'], x['des2'], cfg.METRIC.RATIO,
#                                              x['s1'], x['dot_des1'], x['r_mask1'])
#     ms2 = nearest_neighbor_ratio_match_score(x['des2'], x['des1'], cfg.METRIC.RATIO,
#                                              x['s2'], x['dot_des2'], x['r_mask2'])
#     return (ms1 + ms2) / 2


"""
Evaluation functions
"""


def collect_ids(dot_des, top_k):
    """
    :param dot_des: N x H x W x H x W. Matrix of distances
    :param top_k: Consider if correct match is in k closest values
    """
    n, h, w, _, _ = dot_des.size()
    flat = h * w

    dot_des = dot_des.view(n, flat, flat)

    k_flat = top_k * flat

    ids0 = torch.ones((n, k_flat), dtype=torch.long) * torch.arange(0, n).view(n, 1)
    ids1 = torch.ones((flat, top_k), dtype=torch.long) * torch.arange(0, flat).view(flat, 1)
    ids1 = ids1.view(k_flat, 1).repeat((n, 1))
    # A pair of normalized descriptors will have the maximum value of scalar product if they are the closets
    _, ids2 = dot_des.topk(k=top_k, dim=-1)

    n_flat = n * k_flat

    ids0 = ids0.view(n_flat, 1).to(dot_des.device)
    ids1 = ids1.view(n_flat, 1).to(dot_des.device)
    ids2 = ids2.view(n_flat, 1)
    ids = torch.cat((ids0, ids1, ids2), dim=-1)

    return ids


def get_correct_matches_indices(ids, s, r_mask):
    """
    :param ids: S x 3
    :param s: N x H x W x H x W. Matrix of correspondences
    :param r_mask: N x 1 x H x W
    """
    n, h, w, _, _ = s.size()
    flat = h * w

    s = s.view(n, flat, flat)

    r_mask = r_mask.view(n, 1, flat)
    s = s * r_mask

    matches = s[ids[:, 0], ids[:, 1], ids[:, 2]]
    cm_indices = matches.nonzero().squeeze(-1)

    return cm_indices


def get_unique_matches(cm_ids, r_mask):
    """
    :param cm_ids: S x 3
    :param r_mask: N x 1 x H x W
    """
    n, _, h, w = r_mask.size()
    flat = h * w

    # Consider only unique matches between descriptors from the total number of possible matches
    correct_matches = torch.zeros((flat,)).to(r_mask.device)
    correct_matches[cm_ids[:, 2]] = 1

    correct_matches = correct_matches.sum()
    total = r_mask.sum()

    return correct_matches / total


def nearest_neighbor_match_score(s, dot_des, r_mask, top_k=1):
    """
    :param s: N x H x W x H x W. Matrix of correspondences
    :param dot_des: N x H x W x H x W. Matrix of distances
    :param r_mask: N x 1 x H x W
    :param top_k: Consider if correct match is in k closest values
    """

    ids = collect_ids(dot_des, top_k)
    cm_indices = get_correct_matches_indices(ids, s, r_mask)
    cm_ids = ids[cm_indices]
    nn_ms = get_unique_matches(cm_ids, r_mask)

    return nn_ms


def nearest_neighbor_thresh_match_score(des1, des2, t, s, dot_des, r_mask):
    """
    :param des1: N x C x H x W
    :param des2: N x C x H x W
    :param t: distance threshold
    :param s: N x H x W x H x W. Matrix of correspondences
    :param dot_des: N x H x W x H x W. Matrix of distances
    :param r_mask: N x 1 x H x W
    """
    n, c, h, w = des1.size()
    flat = h * w

    des1 = des1.view(n, c, flat)
    des2 = des2.view(n, c, flat)

    ids = collect_ids(dot_des, 1)
    cm_indices = get_correct_matches_indices(ids, s, r_mask)
    cm_ids = ids[cm_indices]

    # Test correct matches by a threshold

    cm_des1 = des1[cm_ids[:, 0], :, cm_ids[:, 1]]
    cm_des2 = des2[cm_ids[:, 0], :, cm_ids[:, 2]]

    dist = torch.norm(cm_des1 - cm_des2, p=2, dim=1)
    cm_indices = dist.lt(t).nonzero().squeeze(-1)

    cm_ids = cm_ids[cm_indices]
    nnt_ms = get_unique_matches(cm_ids, r_mask)

    return nnt_ms


def nearest_neighbor_ratio_match_score(des1, des2, rt, s, dot_des, r_mask):
    """
   :param des1: N x C x H x W
   :param des2: N x C x H x W
   :param rt: ratio threshold
   :param s: N x H x W x H x W. Matrix of correspondences
   :param dot_des: N x H x W x H x W. Matrix of distances
   :param r_mask: N x 1 x H x W
   """
    n, c, h, w = des1.size()
    flat = h * w

    des1 = des1.view(n, c, flat)
    des2 = des2.view(n, c, flat)

    # Collect first and second closest matches
    ids = collect_ids(dot_des, 2)
    # Determine which of first matches are correct
    cm_indices = get_correct_matches_indices(ids[::2], s, r_mask)
    # Add their corresponding second matches
    i = cm_indices.shape[0]
    cm_indices = cm_indices.repeat(2)
    cm_indices[i:] += 1

    cm_ids = ids[cm_indices]

    cm_des1 = des1[cm_ids[::2, 0], :, cm_ids[::2, 1]]
    cm_des2b = des2[cm_ids[::2, 0], :, cm_ids[::2, 2]]
    cm_des2c = des2[cm_ids[1::2, 0], :, cm_ids[1::2, 2]]

    dist_b = torch.norm(cm_des1 - cm_des2b, p=2, dim=1)
    dist_c = torch.norm(cm_des1 - cm_des2c, p=2, dim=1)

    r_dist = dist_b / dist_c
    cm_indices = r_dist.lt(rt).nonzero().squeeze(-1)

    cm_ids = cm_ids[cm_indices]
    nnr_ms = get_unique_matches(cm_ids, r_mask)

    return nnr_ms


"""
Results visualisation
"""


def torch2cv(img):
    """
    :type img: 1 x C x H x W
    """
    img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)
    return img


def kp2cv(kp):
    """
    :type kp: K x 4
    """
    return cv2.KeyPoint(kp[3], kp[2], 0)


def plot_keypoints(writer, epoch, outputs):
    """
    :param writer: SummaryWriter
    :param epoch: Current train epoch
    :param outputs: list of tuples with all necessary info
    """
    for i, (im1, im2, top_k_mask1, top_k_mask2) in enumerate(outputs):
        im1 = torch2cv(im1)
        im2 = torch2cv(im2)

        kp1 = top_k_mask1.nonzero().cpu().detach().numpy()
        kp2 = top_k_mask2.nonzero().cpu().detach().numpy()

        kp1 = list(map(kp2cv, kp1))
        kp2 = list(map(kp2cv, kp2))

        im1_kp = cv2.drawKeypoints(im1, kp1, None, color=(0, 255, 0))
        im2_kp = cv2.drawKeypoints(im2, kp2, None, color=(0, 255, 0))

        im1_kp = im1_kp.transpose((2, 0, 1))
        im1_kp = torch.from_numpy(im1_kp).unsqueeze(0)

        im2_kp = im2_kp.transpose((2, 0, 1))
        im2_kp = torch.from_numpy(im2_kp).unsqueeze(0)

        writer.add_image(f"s{i}_keypoints", make_grid(torch.cat((im1_kp, im2_kp), dim=0)), epoch)

