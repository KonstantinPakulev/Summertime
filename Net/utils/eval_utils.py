import cv2
import numpy as np
from random import randint

import torch
from torchvision.utils import make_grid

from Net.hpatches_dataset import (S_IMAGE1,
                                  S_IMAGE2)

from Net.utils.common_utils import torch2cv, to_cv2_keypoint, cv2torch, to_cv2_dmatch
from Net.utils.math_utils import calculate_distance_matrix, calculate_inv_similarity_matrix, calculate_inv_similarity_vector
from Net.utils.image_utils import warp_keypoints

# Eval metrics names
LOSS = 'loss'
DET_LOSS = 'det_loss'
DES_LOSS = 'des_loss'
REP_SCORE = 'repeatability_score'
MATCH_SCORE = 'match_score'
NN_MATCH_SCORE = 'nearest_neighbour_match_score'
NNT_MATCH_SCORE = 'nearest_neighbour_thresh_match_score'
NNR_MATCH_SCORE = 'nearest_neighbour_ratio_match_score'
SHOW = 'show'

# Eval metrics inputs
IMAGE_SIZE = 'image_size'
TOP_K = 'top_k'
KP_MT = 'kp_match_thresh'
DS_MT = 'des_match_thresh'
DS_MR = 'des_match_ratio'
KP1 = 'kp1'
KP2 = 'kp2'
W_KP1 = 'w_kp1'
W_KP2 = 'w_kp2'
KP1_DESC = 'desc1'
KP2_DESC = 'desc2'

"""
Mappings to process endpoint and calculate metrics
"""


def l_loss(x):
    return x[LOSS]


def l_det_loss(x):
    return x[DET_LOSS]


def l_des_loss(x):
    return x[DES_LOSS]


def l_rep_score(x):
    return repeatability_score(x[KP1], x[W_KP2], x[KP2], x[TOP_K], x[KP_MT])[0]


def l_match_score(x):
    return match_score(x[KP1], x[W_KP2], x[KP2], x[KP1_DESC], x[KP2_DESC], x[TOP_K], x[KP_MT], x[DS_MT], x[DS_MR])


def l_collect_show(x):
    batch_id = randint(0, x[KP1].size(0) - 1)
    return x[S_IMAGE1][batch_id], x[S_IMAGE2][batch_id], x[KP1][batch_id], x[W_KP2][batch_id], x[KP2][batch_id], \
           x[KP1_DESC][batch_id], x[KP2_DESC][batch_id], x[TOP_K], x[KP_MT]


"""
Evaluation functions
"""


def repeatability_score(kp1, w_kp2, kp2, top_k, kp_match_thresh):
    """
    :param kp1: B x N x 2
    :param w_kp2: B x N x 2
    :param kp2: B x N x 2
    :param top_k: int
    :param kp_match_thresh: float
    """
    _, ids = calculate_distance_matrix(kp1, w_kp2).min(dim=-1)

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, ids, top_k, kp_match_thresh)


def match_score(kp1, w_kp2, kp2, kp1_desc, kp2_desc, top_k, kp_match_thresh, des_match_thresh, des_match_ratio):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param kp1_desc: B x N x C descriptors of kp1
    :param kp2_desc: B x N x C descriptors of kp2
    :param top_k: int
    :param kp_match_thresh: float
    :param des_match_thresh: float
    :param des_match_ratio: float
    """
    _, ids = calculate_inv_similarity_matrix(kp1_desc, kp2_desc).topk(k=2, dim=-1, largest=False)

    nn_match_score, _, _ = nearest_neighbor_match_score(kp1, w_kp2, kp2, ids[:, :, 0],
                                                        top_k, kp_match_thresh)
    nn_thresh_match_score, _, _ = nearest_neighbor_threshold_match_score(kp1, w_kp2, kp2, kp1_desc, kp2_desc,
                                                                         ids[:, :, 0],
                                                                         top_k, kp_match_thresh, des_match_thresh)
    nn_ratio_match_score, _, _ = nearest_neighbor_ratio_match_score(kp1, w_kp2, kp2, kp1_desc, kp2_desc, ids,
                                                                    top_k, kp_match_thresh, des_match_ratio)

    total_match_score = (nn_match_score + nn_thresh_match_score + nn_ratio_match_score) / 3

    return [total_match_score, nn_match_score, nn_thresh_match_score, nn_ratio_match_score]


def nearest_neighbor_match_score(kp1, w_kp2, kp2, ids, top_k, kp_match_thresh):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param ids: B x N; Ids of closest keypoints from kp2 to kp1 by some measure
    :param top_k: int
    :param kp_match_thresh: int
    :return float, B x N1 x 2, B x N2 x 2
    """
    b, n, _ = kp1.size()

    g_ids = ids.unsqueeze(dim=-1).repeat((1, 1, 2))
    w_kp2 = w_kp2.gather(dim=1, index=g_ids)
    kp2 = kp2.gather(dim=1, index=g_ids)

    f_kp1 = kp1.float().view(b * n, 2)
    f_kp2 = w_kp2.float().view(b * n, 2)

    kp1_mask = f_kp1.sum(dim=-1).ne(0.0)
    kp2_mask = f_kp2.sum(dim=-1).ne(0.0)

    dist = torch.pairwise_distance(f_kp1, f_kp2)
    correct_matches = dist.le(kp_match_thresh) * kp1_mask * kp2_mask
    correct_matches = correct_matches.view(b, n, 1).long()

    score = (correct_matches.sum(dim=1).float() / top_k).mean()

    kp1 = kp1 * correct_matches
    kp2 = kp2 * correct_matches

    return score, kp1, kp2


def nearest_neighbor_threshold_match_score(kp1, w_kp2, kp2, kp1_desc, kp2_desc, ids, top_k, kp_match_thresh,
                                           des_match_thresh):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param kp1_desc: B x N x C descriptors of kp1
    :param kp2_desc: B x N x C descriptors of kp2
    :param ids: B x N; Ids of closest keypoints from kp2 to kp1 by some measure
    :param top_k: int
    :param kp_match_thresh: float
    :param des_match_thresh: float
    """
    b, n, c = kp2_desc.size()

    g_ids = ids.unsqueeze(dim=-1).repeat((1, 1, c))
    kp2_desc = kp2_desc.gather(dim=1, index=g_ids)

    f_kp1_desc = kp1_desc.contiguous().view(b * n, c)
    f_kp2_desc = kp2_desc.contiguous().view(b * n, c)

    des_dist = torch.pairwise_distance(f_kp1_desc, f_kp2_desc)
    thresh_mask = des_dist.le(des_match_thresh).unsqueeze(-1).long().view(b, n, 1)  # B x N x 1

    kp1 = kp1 * thresh_mask

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, ids, top_k, kp_match_thresh)


def nearest_neighbor_ratio_match_score(kp1, w_kp2, kp2, kp1_desc, kp2_desc, ids, top_k, kp_match_thresh,
                                       des_match_ratio):
    """
     :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param kp1_desc: B x N x C descriptors of kp1
    :param kp2_desc: B x N x C descriptors of kp2
    :param ids: B x N; Ids of closest keypoints from kp2 to kp1 by some measure
    :param top_k: int
    :param kp_match_thresh: float
    :param des_match_ratio: float
    """
    b, n, c = kp2_desc.size()

    g_ids_b = ids[:, :, 0].unsqueeze(dim=-1).repeat((1, 1, c))
    g_ids_c = ids[:, :, 1].unsqueeze(dim=-1).repeat((1, 1, c))

    f_kp1_desc = kp1_desc.contiguous().view(b * n, c)
    desc2_b = kp2_desc.gather(dim=1, index=g_ids_b).view(b * n, c)
    desc2_c = kp2_desc.gather(dim=1, index=g_ids_c).view(b * n, c)

    des_dist1 = torch.pairwise_distance(f_kp1_desc, desc2_b)
    des_dist2 = torch.pairwise_distance(f_kp1_desc, desc2_c)
    dist_ratio = des_dist1 / des_dist2
    thresh_mask = dist_ratio.le(des_match_ratio).unsqueeze(-1).long().view(b, n, 1)  # B x N x 1

    kp1 = kp1 * thresh_mask

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, ids[:, :, 0], top_k, kp_match_thresh)


"""
Results visualisation
"""


def plot_keypoints(writer, epoch, outputs):
    """
    :param writer: SummaryWriter
    :param epoch: Current train epoch
    :param outputs: list of tuples with all necessary info
    """
    for i, (s_image1, s_image2, kp1, w_kp2, kp2, kp1_desc, kp2_desc, top_k, kp_match_thresh) in enumerate(outputs):
        s_image1 = torch2cv(s_image1)
        s_image2 = torch2cv(s_image2)

        """
        Detected keypoints
        """
        d_kp1 = to_cv2_keypoint(kp1)
        d_kp2 = to_cv2_keypoint(kp2)

        s_image1_kp = cv2.drawKeypoints(s_image1, d_kp1, None, color=(0, 255, 0))
        s_image2_kp = cv2.drawKeypoints(s_image2, d_kp2, None, color=(0, 255, 0))

        s_image1_kp = cv2torch(s_image1_kp).unsqueeze(0)
        s_image2_kp = cv2torch(s_image2_kp).unsqueeze(0)

        """
        Coordinates matched keypoints
        """
        _, km_kp1, km_kp2 = repeatability_score(kp1.unsqueeze(0), w_kp2.unsqueeze(0), kp2.unsqueeze(0),
                                                top_k, kp_match_thresh)

        km_kp1 = to_cv2_keypoint(km_kp1.squeeze(0))
        km_kp2 = to_cv2_keypoint(km_kp2.squeeze(0))

        kp_matches = cv2.drawMatches(s_image1, km_kp1, s_image2, km_kp2, to_cv2_dmatch(km_kp1), None)
        kp_matches = cv2torch(kp_matches)

        """
        Descriptors matched keypoints 
        """
        _, ids = calculate_inv_similarity_matrix(kp1_desc.unsqueeze(0), kp2_desc.unsqueeze(0)).min(dim=-1)
        score, dm_kp1, dm_kp2 = nearest_neighbor_match_score(kp1.unsqueeze(0), w_kp2.unsqueeze(0), kp2.unsqueeze(0),
                                                             ids, top_k, kp_match_thresh)

        dm_kp1 = to_cv2_keypoint(dm_kp1.squeeze(0))
        dm_kp2 = to_cv2_keypoint(dm_kp2.squeeze(0))

        desc_matches = cv2.drawMatches(s_image1, dm_kp1, s_image2, dm_kp2, to_cv2_dmatch(dm_kp1), None)
        desc_matches = cv2torch(desc_matches)

        writer.add_image(f"s{i}_keypoints", make_grid(torch.cat((s_image1_kp, s_image2_kp), dim=0)),
                         epoch)
        writer.add_image(f"s{i}_keypoints_matches", kp_matches, epoch)
        writer.add_image(f"s{i}_descriptor_matches", desc_matches, epoch)
