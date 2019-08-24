import torch
from torchvision.utils import make_grid

from Net.source.utils.common_utils import torch2cv, cv2torch, draw_cv_keypoints, draw_cv_matches
from Net.source.utils.math_utils import calculate_distance_matrix, calculate_inv_similarity_matrix

"""
Evaluation functions
"""


def metric_scores(kp1, w_kp2, kp2, wv_kp2_mask, kp1_desc, kp2_desc, kp_match_thresh, des_match_thresh, des_match_ratio):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param wv_kp2_mask: B x N; w_kp2 on score2 which are visible from score1
    :param kp1_desc: B x N x C descriptors of kp1
    :param kp2_desc: B x N x C descriptors of kp2
    :param kp_match_thresh: float
    :param des_match_thresh: float
    :param des_match_ratio: float
    """
    dnn_values, dnn_ids = calculate_inv_similarity_matrix(kp1_desc, kp2_desc).topk(k=2, dim=-1, largest=False)
    kp_dist = calculate_distance_matrix(kp1, w_kp2)
    knn_values, knn_ids = kp_dist.min(dim=-1)

    b, n, _ = kp1.size()

    dnn_kp_values = torch.zeros((b, n)).to(kp1.device)
    b_ids = torch.arange(kp1.size(0))
    kp_ids = torch.arange(kp1.size(1))
    for i in b_ids:
        dnn_kp_values[i, kp_ids] = kp_dist[i, kp_ids, dnn_ids[i, :, 0]]

    rep_score = abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, None, wv_kp2_mask, knn_values, knn_ids, kp_match_thresh)
    nn_match_score = \
        abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, None, wv_kp2_mask, dnn_kp_values, dnn_ids[:, :, 0], kp_match_thresh)

    thresh_mask = dnn_values[:, :, 0].le(des_match_thresh)  # B x N
    nn_thresh_match_score = \
        abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, thresh_mask, wv_kp2_mask, dnn_kp_values, dnn_ids[:, :, 0], kp_match_thresh)

    dist_ratio = dnn_values[:, :, 0] / dnn_values[:, :, 1]
    ratio_mask = dist_ratio.le(des_match_ratio)  # B x N
    nn_ratio_match_score = \
        abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, ratio_mask, wv_kp2_mask, dnn_kp_values, dnn_ids[:, :, 0], kp_match_thresh)


    return [rep_score, nn_match_score, nn_thresh_match_score, nn_ratio_match_score]


# noinspection PyUnboundLocalVariable
def repeatability_score(kp1, w_kp2, kp2, vw_kp2_mask, kp_match_thresh, provide_kp=False):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param vw_kp2_mask: B x N; w_kp2 on score2 which are visible from score1
    :param kp_match_thresh: int
    :param provide_kp: Also provide keypoints with score
    """
    # Get closest keypoint in w_kp2 for kp1
    nn_values, nn_ids = calculate_distance_matrix(kp1, w_kp2).min(dim=-1)  # B x N

    return abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, None, vw_kp2_mask, nn_values, nn_ids, kp_match_thresh, provide_kp)


# noinspection PyUnboundLocalVariable
def abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, mask1, vw_kp2_mask, nn_values, nn_ids, kp_match_thresh, provide_kp=False):
    """
    :param kp1: B x N x 2; Keypoints from score1
    :param w_kp2: B x N x 2; Keypoints from score2 warped to score1
    :param kp2: B x N x 2; Keypoints from score2
    :param mask1: B x N; Mask on kp1 to remove some points by some criterion
    :param vw_kp2_mask: B x N; w_kp2 on score2 which are visible from score1
    :param nn_values: B x N; Values of distances between kp1 and kp2
    :param nn_ids: B x N; Ids of closest keypoints from kp2 to kp1 by some measure
    :param kp_match_thresh: int
    :param provide_kp: Also provide keypoints with score
    """
    b, n, _ = w_kp2.size()
    b_ids = torch.arange(b).to(kp1.device)
    n_ids = torch.arange(n).to(kp1.device)

    matched_kp1 = nn_values.lt(kp_match_thresh)
    if mask1 is not None:
        matched_kp1 = matched_kp1 * mask1
    matches_kp2 = torch.zeros((b, n), dtype=torch.uint8).to(kp2.device)

    if provide_kp:
        matched_kp2_dist = torch.ones((b, n)).to(kp2.device) * kp_match_thresh
        matched_kp2_ids = torch.zeros((b, n), dtype=torch.long).to(kp2.device)

    for i in b_ids:
        for j in n_ids:
                if provide_kp:
                    if matched_kp1[i, j] and nn_values[i, j] < matched_kp2_dist[i, nn_ids[i, j]]:
                        matched_kp2_dist[i, nn_ids[i, j]] = nn_values[i, j]
                        matched_kp2_ids[i, nn_ids[i, j]] = j
                else:
                    matches_kp2[i, nn_ids[i, j]] = torch.max(matched_kp1[i, j], matches_kp2[i, nn_ids[i, j]])

    if provide_kp:
        matches_kp2 = matched_kp2_dist.lt(kp_match_thresh)

    matched_kp2 = matches_kp2 * vw_kp2_mask
    score = (matched_kp2.sum(dim=1).float() / vw_kp2_mask.sum(dim=1).float()).mean()

    if provide_kp:
        assert b == 1

        kp1_mask = torch.zeros((b, n), dtype=torch.uint8).to(kp1.device)
        for i in b_ids:
            for j in n_ids:
                if matched_kp2_ids[i, nn_ids[i, j]] == j and vw_kp2_mask[i, nn_ids[i, j]]:
                    kp1_mask[i, j] = 1

        matched_kp1 = matched_kp1 * kp1_mask
        matched_kp1 = matched_kp1.unsqueeze(-1).long()

        kp2 = kp2[0].index_select(dim=0, index=nn_ids[0])

        return score, kp1 * matched_kp1, kp2 * matched_kp1
    else:
        return score


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
    for i, (s_image1, s_image2, kp1, w_kp2, kp2, vw_kp2_mask, kp1_desc, kp2_desc) in enumerate(outputs):
        s_image1 = torch2cv(s_image1)
        s_image2 = torch2cv(s_image2)

        """
        Detected keypoints
        """
        s_image1_kp = cv2torch(draw_cv_keypoints(s_image1, kp1.squeeze(0), (0, 255, 0))).unsqueeze(0)
        s_image2_kp = cv2torch(draw_cv_keypoints(s_image2, kp2.squeeze(0), (0, 255, 0))).unsqueeze(0)

        """
        Coordinates matched keypoints
        """
        kp_dist = calculate_distance_matrix(kp1, w_kp2)
        knn_values, knn_ids = kp_dist.min(dim=-1)
        _, km_kp1, km_kp2 = abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2,
                                                                  None, vw_kp2_mask, knn_values, knn_ids,
                                                                  kp_match_thresh, True)

        kp_matches = cv2torch(draw_cv_matches(s_image1, s_image2, km_kp1.squeeze(0), km_kp2.squeeze(0)))

        writer.add_image(f"s{i}_keypoints", make_grid(torch.cat((s_image1_kp, s_image2_kp), dim=0)), epoch)
        writer.add_image(f"s{i}_keypoints_matches", kp_matches, epoch)

        """
        Descriptors matched keypoints 
        """
        if kp1_desc is not None and kp2_desc is not None:
            _, dnn_ids = calculate_inv_similarity_matrix(kp1_desc, kp2_desc).min(dim=-1)
            dnn_kp_values = kp_dist[0, torch.arange(kp1_desc.size(1)), dnn_ids]

            _, dm_kp1, dm_kp2 = abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2,
                                                                      None, vw_kp2_mask, dnn_kp_values, dnn_ids,
                                                                      kp_match_thresh, True)

            desc_matches = cv2torch(draw_cv_matches(s_image1, s_image2, dm_kp1.squeeze(0), dm_kp2.squeeze(0)))

            writer.add_image(f"s{i}_descriptor_matches", desc_matches, epoch)
