from enum import Enum

import torch

from Net.source.utils.math_utils import calculate_distance_mat, inv_cos_sim_mat


def get_gt_matches(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh, return_reproj=False):
    """
    :param kp1: B x N x 2
    :param kp2: B x N x 2
    :param w_kp1: B x N x 2
    :param w_kp2: B x N x 2
    :param w_vis_kp1_mask: B x N
    :param w_vis_kp2_mask: B x N
    :param px_thresh: int
    :param return_reproj: bool
    """
    mutual_gt_matches_mask1, nn_kp_values1, nn_kp_ids1 = \
        get_mutual_gt_matches(w_kp1, kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)

    mutual_gt_matches_mask2, nn_kp_values2, nn_kp_ids2 = \
        get_mutual_gt_matches(kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)

    mutual_gt_matches_mask = mutual_gt_matches_mask1 * mutual_gt_matches_mask2

    nn_kp_values, nn_kp_values_ids = torch.cat([nn_kp_values1.unsqueeze(-1),
                                                nn_kp_values2.unsqueeze(-1)], dim=-1).min(dim=-1)

    nn_kp_ids = torch.cat([nn_kp_ids1.unsqueeze(-1),
                           nn_kp_ids2.unsqueeze(-1)], dim=-1)

    nn_kp_ids = torch.gather(nn_kp_ids, dim=-1, index=nn_kp_values_ids.unsqueeze(-1)).squeeze(-1)

    if return_reproj:
        return mutual_gt_matches_mask, nn_kp_values, nn_kp_ids

    else:
        return mutual_gt_matches_mask, nn_kp_ids


def get_mutual_gt_matches(w_kp1, kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh):
    kp_dist1 = calculate_distance_mat(w_kp1, kp2)
    kp_dist1 = mask_non_visible_pairs(kp_dist1, w_vis_kp1_mask, w_vis_kp2_mask)

    nn_kp_values1, nn_kp_ids1 = kp_dist1.min(dim=-1)
    _, nn_kp_ids2 = kp_dist1.min(dim=-2)

    mutual_gt_matches_mask = get_mutual_matches(nn_kp_ids1, nn_kp_ids2) * nn_kp_values1.le(px_thresh)

    return mutual_gt_matches_mask, nn_kp_values1, nn_kp_ids1


def get_mutual_desc_matches(kp1_desc, kp2_desc, dd_measure, lowe_ratio=None):
    desc_dist = calculate_descriptor_distance(kp1_desc, kp2_desc, dd_measure)

    if lowe_ratio is not None:
        nn_desc_value1, nn_desc_ids1 = desc_dist.topk(dim=-1, k=2, largest=False)
        nn_desc_value2, nn_desc_ids2 = desc_dist.topk(dim=-2, k=2, largest=False)

        mutual_desc_matches_mask = get_mutual_matches(nn_desc_ids1[..., 0], nn_desc_ids2[:, 0, :])

        # Create Lowe ratio test masks
        lowe_ratio_mask1 = nn_desc_value1[..., 0] < nn_desc_value1[..., 1] * lowe_ratio
        lowe_ratio_mask2 = nn_desc_value2[:, 0, :] < nn_desc_value2[:, 1, :] * lowe_ratio

        nn_lowe_ratio_mask2 = torch.gather(lowe_ratio_mask2, -1, nn_desc_ids1[..., 0])

        mutual_desc_matches_mask *= lowe_ratio_mask1 * nn_lowe_ratio_mask2

        return mutual_desc_matches_mask, nn_desc_ids1[..., 0]
    else:
        nn_desc_ids1 = desc_dist.min(dim=-1)[1]
        nn_desc_ids2 = desc_dist.min(dim=-2)[1]

        mutual_desc_matches_mask = get_mutual_matches(nn_desc_ids1, nn_desc_ids2)

        return mutual_desc_matches_mask, nn_desc_ids1


def verify_mutual_desc_matches(nn_ids, kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh, return_reproj=False):
    kp_dist1 = calculate_distance_mat(w_kp1, kp2)
    kp_dist1 = mask_non_visible_pairs(kp_dist1, w_vis_kp1_mask, w_vis_kp2_mask)

    kp_dist2 = calculate_distance_mat(kp1, w_kp2)
    kp_dist2 = mask_non_visible_pairs(kp_dist2, w_vis_kp1_mask, w_vis_kp2_mask)

    # Retrieve correspondent keypoints distances
    nn_kp_values1 = torch.gather(kp_dist1, -1, nn_ids.unsqueeze(-1))
    nn_kp_values2 = torch.gather(kp_dist2, -1, nn_ids.unsqueeze(-1))

    # Retrieve minimum distance among two re-projections
    nn_kp_values = torch.cat([nn_kp_values1, nn_kp_values2], dim=-1).min(dim=-1)[0]

    v_mutual_desc_matches_mask = nn_kp_values.le(px_thresh)

    if return_reproj:
        return v_mutual_desc_matches_mask, nn_kp_values

    else:
        return v_mutual_desc_matches_mask


"""
Support functions
"""

# TODO. Replace with str from config
class DescriptorDistance(Enum):
    INV_COS_SIM = 0
    L2 = 1


# TODO. Rework
def calculate_descriptor_distance(kp1_desc, kp2_desc, dd_measure):
    return inv_cos_sim_mat(kp1_desc, kp2_desc)
    # if dd_measure == DescriptorDistance.INV_COS_SIM:
    #
    # elif dd_measure == DescriptorDistance.L2:
    #     return calculate_distance_mat(kp1_desc, kp2_desc)
    # else:
    #     return None


def mask_non_visible_pairs(dist_matrix, wv_kp1_mask, wv_kp2_mask):
    """
    :param dist_matrix: B x N1 x N2
    :param wv_kp1_mask: B x N1
    :param wv_kp2_mask: B x N2
    """
    b, n1, n2 = dist_matrix.shape

    max_dist = dist_matrix.max() * 2

    # Ensure only keypoints in a shared region are considered
    dist_matrix += (1 - wv_kp1_mask.float().view(b, n1, 1)) * max_dist
    dist_matrix += (1 - wv_kp2_mask.float().view(b, 1, n2)) * max_dist

    return dist_matrix


def get_mutual_matches(nn_ids1, nn_ids2):
    ids = torch.arange(0, nn_ids1.shape[1]).repeat(nn_ids1.shape[0], 1).to(nn_ids1.device)
    nn_ids = torch.gather(nn_ids2, -1, nn_ids1)

    return ids == nn_ids


def select_kp(kp, ids):
    return torch.gather(kp, 1, ids.unsqueeze(-1).repeat(1, 1, 2))


def get_num_vis_gt_matches(w_vis_kp1_mask, w_vis_kp2_mask):
    v1 = w_vis_kp1_mask.sum(dim=-1).unsqueeze(-1)
    v2 = w_vis_kp2_mask.sum(dim=-1).unsqueeze(-1)
    num_vis_gt_matches = torch.cat([v1, v2], dim=-1).min(dim=-1)[0].float().clamp(min=1e-8)

    return num_vis_gt_matches


# Legacy code

# def get_mutual_gt_matches_old(w_kp1, kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh):
#     kp_dist1 = calculate_distance_mat(w_kp1, kp2)
#     kp_dist1 = mask_non_visible_pairs(kp_dist1, w_vis_kp1_mask, w_vis_kp2_mask)
#
#     nn_kp_values1, nn_kp_ids1 = kp_dist1.min(dim=-1)
#     _, nn_kp_ids2 = kp_dist1.min(dim=-2)
#
#     mutual_gt_matches_mask = get_mutual_matches(nn_kp_ids1, nn_kp_ids2)
#     threshold_mask = get_threshold_mask(nn_kp_values1, px_thresh)
#
#     mutual_gt_matches_mask = mutual_gt_matches_mask * threshold_mask
#
#     return mutual_gt_matches_mask, nn_kp_values1, nn_kp_ids1


# def get_best_gt_matches_old(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh, mutual=True):
#
#     if mutual:
#         gt_matches_mask, _, nn_kp_ids1 = \
#             get_mutual_gt_matches_old(w_kp1, kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)
#
#         gt_matches_mask2, _, nn_kp_ids2 = \
#             get_mutual_gt_matches_old(kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)
#     else:
#         gt_matches_mask, nn_kp_ids1 = \
#             get_gt_matches_old(w_kp1, kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)
#
#         gt_matches_mask2, nn_kp_ids2 = \
#             get_gt_matches_old(kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)
#
#     gt_matches_count = gt_matches_mask.float().sum(dim=-1).mean(dim=0)
#     gt_matches_count2 = gt_matches_mask2.float().sum(dim=-1).mean(dim=0)
#
#     replace_mask = gt_matches_count < gt_matches_count2
#
#     gt_matches_mask[:, replace_mask, ...] = gt_matches_mask2[:, replace_mask, ...]
#     nn_kp_ids1[replace_mask, ...] = nn_kp_ids2[replace_mask, ...]
#
#     return gt_matches_mask, nn_kp_ids1


# def get_gt_matches_old(w_kp1, kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh):
#     kp_dist1 = calculate_distance_mat(w_kp1, kp2)
#     kp_dist1 = mask_non_visible_pairs(kp_dist1, w_vis_kp1_mask, w_vis_kp2_mask)
#
#     nn_kp_values1, nn_kp_ids1 = kp_dist1.min(dim=-1)
#
#     unique_mask = calculate_unique_match_mask(nn_kp_values1, nn_kp_ids1)
#     threshold_mask = get_threshold_mask(nn_kp_values1, px_thresh)
#
#     gt_matches_mask = threshold_mask * unique_mask
#
#     return gt_matches_mask, nn_kp_ids1


# def get_mutual_desc_matches_v2(kp1, kp2, desc1, desc2, kp1_desc, kp2_desc, grid_size, dd_measure, lowe_ratio=None):
#     desc_dist = calculate_descriptor_distance(kp1_desc, kp2_desc, dd_measure)
#
#     nn_desc_value1, nn_desc_ids1 = desc_dist.topk(dim=-1, k=2, largest=False)
#     nn_desc_value2, nn_desc_ids2 = desc_dist.topk(dim=-2, k=2, largest=False)
#
#     mutual_desc_matches_mask = get_mutual_matches(nn_desc_ids1[..., 0], nn_desc_ids2[:, 0, :])
#
#     # Create Lowe ratio test masks
#     lowe_ratio_mask1 = nn_desc_value1[..., 0] < nn_desc_value1[..., 1] * lowe_ratio
#     lowe_ratio_mask2 = nn_desc_value2[:, 0, :] < nn_desc_value2[:, 1, :] * lowe_ratio
#
#     nn_lowe_ratio_mask2 = torch.gather(lowe_ratio_mask2, -1, nn_desc_ids1[..., 0])
#
#     mutual_desc_matches_mask *= lowe_ratio_mask1 * nn_lowe_ratio_mask2
#
#     # Gather neighbouring descriptors and stack them with keypoints descriptors
#
#     neigh_desc1 = sample_neigh_desc(desc1, kp1, grid_size)
#
#     neigh_desc2 = sample_neigh_desc(desc2, kp2, grid_size)
#     nn_neigh_desc2 = torch.gather(neigh_desc2, 2, nn_desc_ids1[..., 0].unsqueeze(1).unsqueeze(-1).repeat(1, 8, 1, 64))
#
#     second_mask = inv_cos_sim_vec(neigh_desc1.view(-1, 512 * 8, 64), nn_neigh_desc2.view(-1, 512 * 8, 64)) < 0.25
#     second_mask = second_mask.view(-1, 8, 512).sum(dim=1) > 5
#
#     # print(mutual_desc_matches_mask.sum(dim=-1))
#     # print(second_mask.sum(dim=-1))
#
#     return mutual_desc_matches_mask * second_mask, nn_desc_ids1[..., 0]
# def calculate_unique_match_mask(values, ids):
#     """
#     :param values: B x N, measure of closeness for each match
#     :param ids: B x N, matching ids
#     """
#     # Get permutation of nn_kp_ids according to decrease of nn_kp_values
#     nn_perm = values.argsort(dim=-1, descending=True)
#     perm_nn_kp_ids = torch.gather(ids, -1, nn_perm)
#
#     # Remove duplicate matches in each scene
#     unique_match_mask = torch.zeros_like(values, dtype=torch.bool).to(values.device)
#
#     for i, b_nn_kp_ids in enumerate(perm_nn_kp_ids):
#         # Find unique elements in each scene
#         b_unique, b_inv_indices = torch.unique(b_nn_kp_ids, sorted=False, return_inverse=True)
#
#         # Restore forward mapping
#         b_indices = torch.zeros_like(b_unique)
#         b_indices[b_inv_indices] = nn_perm[i]
#
#         # Create unique match mask
#         unique_match_mask[i][b_indices] = True
#
#     return unique_match_mask
# def get_threshold_mask(nn_values, px_thresh):
#     thresh_mask = torch.zeros(len(px_thresh), nn_values.shape[0], nn_values.shape[1],
#                               dtype=torch.bool).to(nn_values.device)
#
#     for i, thresh in enumerate(px_thresh):
#         thresh_mask[i] = nn_values.le(thresh)
#
#     return thresh_mask