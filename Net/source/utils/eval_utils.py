import cv2
import numpy as np

import torch
from torchvision.utils import make_grid

from Net.source.utils.common_utils import torch2cv, cv2torch, draw_cv_keypoints, draw_cv_matches
from Net.source.utils.math_utils import calculate_distance_matrix, calculate_inv_similarity_matrix

"""
Evaluation functions
"""


def repeatability_score(kp1, w_kp2, kp2, visible1, kp_match_thresh):
    """
    :param kp1: B x N x 2
    :param w_kp2: B x N x 2
    :param kp2: B x N x 2
    :param visible1: B x N
    :param kp_match_thresh: float
    """
    _, ids = calculate_distance_matrix(kp1, w_kp2).min(dim=-1)

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, visible1, ids, kp_match_thresh)


def match_score(kp1, w_kp2, kp2, visible1, kp1_desc, kp2_desc, kp_match_thresh, des_match_thresh, des_match_ratio):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param visible1: B x N
    :param kp1_desc: B x N x C descriptors of kp1
    :param kp2_desc: B x N x C descriptors of kp2
    :param kp_match_thresh: float
    :param des_match_thresh: float
    :param des_match_ratio: float
    """
    _, ids = calculate_inv_similarity_matrix(kp1_desc, kp2_desc).topk(k=2, dim=-1, largest=False)

    nn_match_score, _, _ = nearest_neighbor_match_score(kp1, w_kp2, kp2, visible1, ids[:, :, 0], kp_match_thresh)
    nn_thresh_match_score, _, _ = nearest_neighbor_threshold_match_score(kp1, w_kp2, kp2, visible1, kp1_desc, kp2_desc,
                                                                         ids[:, :, 0], kp_match_thresh, des_match_thresh)
    nn_ratio_match_score, _, _ = nearest_neighbor_ratio_match_score(kp1, w_kp2, kp2, visible1, kp1_desc, kp2_desc, ids,
                                                                    kp_match_thresh, des_match_ratio)

    total_match_score = (nn_match_score + nn_thresh_match_score + nn_ratio_match_score) / 3

    return [total_match_score, nn_match_score, nn_thresh_match_score, nn_ratio_match_score]


def nearest_neighbor_match_score(kp1, w_kp2, kp2, visible1, ids, kp_match_thresh):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param visible1: B x N
    :param ids: B x N; Ids of closest keypoints from kp2 to kp1 by some measure
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
    correct_matches = correct_matches.view(b, n, 1)

    kp1 = kp1 * correct_matches.long()
    kp2 = kp2 * correct_matches.long()

    if visible1 is not None:
        score = (correct_matches.squeeze(-1).float().sum(dim=-1) / visible1.sum(dim=1).float()).mean()
        return score, kp1, kp2
    else:
        return kp1, kp2


def nearest_neighbor_threshold_match_score(kp1, w_kp2, kp2, visible1, kp1_desc, kp2_desc, ids, kp_match_thresh,
                                           des_match_thresh):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param visible1: B x N
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

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, visible1, ids, kp_match_thresh)


def nearest_neighbor_ratio_match_score(kp1, w_kp2, kp2, visible, kp1_desc, kp2_desc, ids, kp_match_thresh,
                                       des_match_ratio):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param visible: visibility mask
    :param kp1_desc: B x N x C descriptors of kp1
    :param kp2_desc: B x N x C descriptors of kp2
    :param ids: B x N; Ids of closest keypoints from kp2 to kp1 by some measure
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

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, visible, ids[:, :, 0], kp_match_thresh)


"""
Results visualisation
"""


def plot_keypoints_and_descriptors(writer, epoch, outputs, kp_match_thresh):
    """
    :param writer: SummaryWriter
    :param epoch: Current train epoch
    :param outputs: list of tuples with all necessary info
    :param kp_match_thresh: float
    """
    for i, (s_image1, s_image2, kp1, w_kp2, kp2, kp1_desc, kp2_desc) in enumerate(outputs):
        s_image1 = torch2cv(s_image1)
        s_image2 = torch2cv(s_image2)

        """
        Detected keypoints
        """
        s_image1_kp = cv2torch(draw_cv_keypoints(s_image1, kp1, (0, 255, 0))).unsqueeze(0)
        s_image2_kp = cv2torch(draw_cv_keypoints(s_image2, kp2, (0, 255, 0))).unsqueeze(0)

        """
        Coordinates matched keypoints
        """
        km_kp1, km_kp2 = repeatability_score(kp1.unsqueeze(0), w_kp2.unsqueeze(0), kp2.unsqueeze(0), None, kp_match_thresh)

        kp_matches = cv2torch(draw_cv_matches(s_image1, s_image2, km_kp1.squeeze(0), km_kp2.squeeze(0)))

        writer.add_image(f"s{i}_keypoints", make_grid(torch.cat((s_image1_kp, s_image2_kp), dim=0)),
                         epoch)
        writer.add_image(f"s{i}_keypoints_matches", kp_matches, epoch)

        """
        Descriptors matched keypoints 
        """
        if kp1_desc is not None and kp2_desc is not None:
            _, ids = calculate_inv_similarity_matrix(kp1_desc.unsqueeze(0), kp2_desc.unsqueeze(0)).min(dim=-1)
            dm_kp1, dm_kp2 = nearest_neighbor_match_score(kp1.unsqueeze(0), w_kp2.unsqueeze(0), kp2.unsqueeze(0),
                                                          None, ids, kp_match_thresh)

            desc_matches = cv2torch(draw_cv_matches(s_image1, s_image2, dm_kp1.squeeze(0), dm_kp2.squeeze(0)))

            writer.add_image(f"s{i}_descriptor_matches", desc_matches, epoch)
