import numpy as np

import torch

from Net.source.utils.math_utils import revert_data_transform, compose_gt_transform, epipolar_distance, \
    change_intrinsics, get_gt_rel_pose, angle_mat, angle_vec, E_param

from Net.source.utils.matching_utils import get_gt_matches, get_mutual_desc_matches, select_kp, \
    verify_mutual_desc_matches, get_num_vis_gt_matches

from Net.source.utils.pose_utils import prepare_rel_pose, prepare_param_rel_pose


"""
Metrics names
"""

REP = 'rep'
REP_NUM_MATCHES = 'rep_num_matches'
REP_NUM_VIS_GT_MATCHES = 'rep_num_vis_gt_matches'

MS = 'ms'
MS_NUM_MATCHES = 'ms_num_matches'
MS_NUM_VIS_GT_MATCHES = 'ms_num_vis_gt_matches'

MMA = 'mma'
MMA_NUM_MATCHES = 'mma_num_matches'
MMA_NUM_VIS_GT_MATCHES = 'mma_num_vis_gt_matches'

EMS = 'ems'
EMS_NUM_MATCHES = 'ems_num_matches'
EMS_NUM_VIS_GT_MATCHES = 'ems_num_vis_gt_matches'

REL_POSE = 'rel_pose'
R_ERR = 'r_err'
T_ERR = 't_err'
NUM_INL = 'num_inl'

PARAM_REL_POSE = 'param_rel_pose'
R_PARAM_ERR = 'r_param_err'
T_PARAM_ERR = 't_param_err'
SUCCESS_MASK = 'success_mask'


"""
Metric functions
"""

# TODO. Seems like there is some issue with enumerator. Fix

def repeatability_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh, detailed):
    """
    :param w_kp1: B x N x 2; keypoints on the first image projected to the second
    :param kp2: B x N x 2; keypoints on the second image
    :param w_vis_kp1_mask: B x N; keypoints on the first image which are visible on the second
    :param w_vis_kp2_mask: B x N; keypoints on the second image which are visible on the first
    :param px_thresh: P; torch.tensor
    :param detailed: bool
    :return P or P x B, P x B, B x N, B x N, P x B x N
    """
    # Use the largest px threshold to determine matches
    gt_matches_mask, nn_kp_values, nn_kp_ids = get_gt_matches(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                                                              px_thresh[-1], return_reproj=True)

    # Select minimum number of visible points for each scene
    num_vis_gt_matches = get_num_vis_gt_matches(w_vis_kp1_mask, w_vis_kp2_mask)

    num_thresh = len(px_thresh)
    b, n = kp1.shape[:2]

    if detailed:
        rep_scores = torch.zeros(num_thresh, b)
        num_matches = torch.zeros(num_thresh, b)
        match_mask = torch.zeros(num_thresh, b, n)
    else:
        rep_scores = torch.zeros(num_thresh)

    # Filter matches by lower thresholds
    for i, thresh in enumerate(px_thresh):
        if i != num_thresh - 1:
            i_gt_matches_mask = nn_kp_values.le(thresh)
        else:
            i_gt_matches_mask = gt_matches_mask

        i_num_matches = i_gt_matches_mask.sum(dim=-1).float()

        if detailed:
            rep_scores[i] = i_num_matches / num_vis_gt_matches
            num_matches[i] = i_num_matches
            match_mask[i] = i_gt_matches_mask
        else:
            rep_scores[i] = (i_num_matches / num_vis_gt_matches).mean()

    if detailed:
        return rep_scores, num_matches, num_vis_gt_matches, nn_kp_ids, match_mask
    else:
        return rep_scores


def match_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, kp1_desc, kp2_desc,
                px_thresh, dd_measure, detailed=False):
    """
    :param kp1: B x N x 2; keypoints on the first image
    :param w_kp1: B x N x 2; keypoints on the first image projected to the second
    :param kp2: B x N x 2; keypoints on the second image
    :param w_kp2: B x N x 2; keypoints on the second image projected to the first
    :param w_vis_kp1_mask: B x N; keypoints on the first image which are visible on the second
    :param w_vis_kp2_mask: B x N; keypoints on the second image which are visible on the first
    :param kp1_desc: B x N x C; descriptors for keypoints on the first image
    :param kp2_desc: B x N x C; descriptors for keypoints on the second image
    :param px_thresh: P; list
    :param dd_measure: measure of descriptor distance. Can be L2-norm or similarity measure
    :param detailed: return detailed information :type bool
    """
    mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, dd_measure, None)

    # Verify matches by using the largest pixel threshold
    v_mutual_desc_matches_mask, nn_kp_values = verify_mutual_desc_matches(nn_desc_ids, kp1, kp2, w_kp1, w_kp2,
                                                                          w_vis_kp1_mask, w_vis_kp2_mask, px_thresh[-1],
                                                                          return_reproj=True)

    # Select minimum number of visible points for each scene
    num_vis_gt_matches = get_num_vis_gt_matches(w_vis_kp1_mask, w_vis_kp2_mask)

    num_thresh = len(px_thresh)
    b, n = kp1.shape[:2]

    if detailed:
        m_scores = torch.zeros(num_thresh, b)
        num_matches = torch.zeros(num_thresh, b)
        match_mask = torch.zeros(num_thresh, b, n)
    else:
        m_scores = torch.zeros(num_thresh)

    # Filter matches by lower thresholds
    for i, thresh in enumerate(px_thresh):
        if i != num_thresh - 1:
            i_mutual_matches_mask = mutual_desc_matches_mask * nn_kp_values.le(thresh)
        else:
            i_mutual_matches_mask = mutual_desc_matches_mask * v_mutual_desc_matches_mask

        i_num_matches = i_mutual_matches_mask.sum(dim=-1).float()

        if detailed:
            m_scores[i] = i_num_matches / num_vis_gt_matches
            num_matches[i] = i_num_matches
            match_mask[i] = i_mutual_matches_mask
        else:
            m_scores[i] = (i_num_matches / num_vis_gt_matches).mean()

    if detailed:
        return m_scores, num_matches, num_vis_gt_matches, nn_desc_ids, match_mask
    else:
        return m_scores


def mean_matching_accuracy(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, kp1_desc, kp2_desc,
                           px_thresh, dd_measure, detailed=False):
    """
    :param kp1: B x N x 2; keypoints on the first image
    :param w_kp1: B x N x 2; keypoints on the first image projected to the second
    :param kp2: B x N x 2; keypoints on the second image
    :param w_kp2: B x N x 2; keypoints on the second image projected to the first
    :param w_vis_kp1_mask: B x N; keypoints on the first image which are visible on the second
    :param w_vis_kp2_mask: B x N; keypoints on the second image which are visible on the first
    :param kp1_desc: B x N x C; descriptors for keypoints on the first image
    :param kp2_desc: B x N x C; descriptors for keypoints on the second image
    :param px_thresh: list; keypoints distance thresholds
    :param dd_measure: measure of descriptor distance. Can be L2-norm or similarity measure
    :param detailed: return detailed information :type bool
    """
    mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, dd_measure, None)

    # Verify matches by using the largest pixel threshold
    v_mutual_desc_matches_mask, nn_kp_values = verify_mutual_desc_matches(nn_desc_ids, kp1, kp2, w_kp1, w_kp2,
                                                                          w_vis_kp1_mask, w_vis_kp2_mask, px_thresh[-1],
                                                                          return_reproj=True)

    num_vis_gt_matches = mutual_desc_matches_mask.sum(dim=-1).float().clamp(min=1e-8)

    num_thresh = len(px_thresh)
    b, n = kp1.shape[:2]

    if detailed:
        mma_scores = torch.zeros(num_thresh, b)
        num_matches = torch.zeros(num_thresh, b)
        match_mask = torch.zeros(num_thresh, b, n)
    else:
        mma_scores = torch.zeros(num_thresh)

    for i, thresh in enumerate(px_thresh):
        if i != num_thresh - 1:
            i_mutual_matches_mask = mutual_desc_matches_mask * nn_kp_values.le(thresh)
        else:
            i_mutual_matches_mask = mutual_desc_matches_mask * v_mutual_desc_matches_mask

        i_num_matches = i_mutual_matches_mask.sum(dim=-1).float()

        if detailed:
            mma_scores[i] = i_num_matches / num_vis_gt_matches
            num_matches[i] = i_num_matches
            match_mask[i] = i_mutual_matches_mask
        else:
            mma_scores[i] = (i_num_matches / num_vis_gt_matches).mean()

    if detailed:
        return mma_scores, num_matches, num_vis_gt_matches, nn_desc_ids, match_mask
    else:
        return mma_scores


def epipolar_match_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                         kp1_desc, kp2_desc, shift_scale1, shift_scale2,
                         intrinsics1, intrinsics2, extrinsics1, extrinsics2,
                         px_thresh, dd_measure, detailed=False):
    mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, dd_measure, None)

    # Verify matches by using the largest pixel threshold
    v_mutual_desc_matches_mask, nn_kp_values = verify_mutual_desc_matches(nn_desc_ids, kp1, kp2, w_kp1, w_kp2,
                                                                          w_vis_kp1_mask, w_vis_kp2_mask, px_thresh[-1], return_reproj=True)

    # Select minimum number of visible points for each scene
    num_vis_gt_matches = get_num_vis_gt_matches(w_vis_kp1_mask, w_vis_kp2_mask)

    num_thresh = len(px_thresh)
    b, n = kp1.shape[:2]

    o_kp1 = revert_data_transform(kp1, shift_scale1)
    o_kp2 = revert_data_transform(kp2, shift_scale2)

    nn_o_kp2 = select_kp(o_kp2, nn_desc_ids)

    F = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2)

    ep_dist = epipolar_distance(o_kp1, nn_o_kp2, F)

    if detailed:
        em_scores = torch.zeros(num_thresh, b)
        num_matches = torch.zeros(num_thresh, b)
        match_mask = torch.zeros(num_thresh, b, n)
    else:
        em_scores = torch.zeros(num_thresh)

    for i, thresh in enumerate(px_thresh):
        if i != num_thresh - 1:
            i_mutual_matches_mask = mutual_desc_matches_mask * nn_kp_values.le(thresh) * ep_dist.le(thresh)
        else:
            i_mutual_matches_mask = mutual_desc_matches_mask * v_mutual_desc_matches_mask * ep_dist.le(thresh)

        i_num_matches = i_mutual_matches_mask.sum(dim=-1).float()

        if detailed:
            em_scores[i] = i_num_matches / num_vis_gt_matches
            num_matches[i] = i_num_matches
            match_mask[i] = i_mutual_matches_mask
        else:
            em_scores[i] = (i_num_matches / num_vis_gt_matches).mean()

    if detailed:
        return em_scores, num_matches, num_vis_gt_matches, nn_desc_ids, match_mask
    else:
        return em_scores


def relative_pose_error(kp1, kp2, kp1_desc, kp2_desc, shift_scale1, shift_scale2, intrinsics1, intrinsics2,
                        extrinsics1, extrinsics2, px_thresh, detailed=False):
    """
    :param kp1: B x N x 2
    :param kp2: B x N x 2
    :param kp1_desc: B x N x C
    :param kp2_desc: B x N x C
    :param shift_scale1: B x 4
    :param shift_scale2: B x 4
    :param intrinsics1: B x 3 x 3
    :param intrinsics2: B x 3 x 3
    :param extrinsics1: B x 4 x 4
    :param extrinsics2: B x 4 x 4
    :param px_thresh: list
    :param detailed: bool
    """
    mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, None, 0.9)

    r_kp1 = revert_data_transform(kp1, shift_scale1)
    r_kp2 = revert_data_transform(kp2, shift_scale2)

    nn_r_kp2 = select_kp(r_kp2, nn_desc_ids)

    num_thresh = len(px_thresh)
    b, n = kp1.shape[:2]

    gt_rel_pose = get_gt_rel_pose(extrinsics1, extrinsics2)

    R_err = torch.zeros(num_thresh, b)
    t_err = torch.zeros(num_thresh, b)

    est_inl_mask = torch.zeros(num_thresh, b, n, dtype=torch.bool).to(kp1.device)

    for i, thresh in enumerate(px_thresh):
        i_est_rel_pose, i_est_inl_mask = prepare_rel_pose(r_kp1, nn_r_kp2, mutual_desc_matches_mask,
                                                          intrinsics1, intrinsics2, thresh)

        R_err[i] = angle_mat(i_est_rel_pose[:, :3, :3], gt_rel_pose[:, :3, :3])
        t_err[i] = angle_vec(i_est_rel_pose[:, :3, 3], gt_rel_pose[:, :3, 3])

    if detailed:
        return R_err, t_err, est_inl_mask, mutual_desc_matches_mask, nn_desc_ids

    else:
        return R_err, t_err, est_inl_mask


def relative_param_pose_error(kp1, kp2, kp1_desc, kp2_desc, shift_scale1, shift_scale2,
                              intrinsics1, intrinsics2, extrinsics1, extrinsics2, px_thresh):
    mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, None, 0.9)

    r_kp1 = revert_data_transform(kp1, shift_scale1)
    r_kp2 = revert_data_transform(kp2, shift_scale2)

    nn_r_kp2 = select_kp(r_kp2, nn_desc_ids)
    nn_r_i1_kp2 = change_intrinsics(nn_r_kp2, intrinsics2, intrinsics1)

    gt_E_param = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2, E_param)

    num_thresh = len(px_thresh)
    b = kp1.shape[0]

    R_param_err = torch.zeros(num_thresh, b)
    t_param_err = torch.zeros(num_thresh, b)

    success_mask = torch.zeros(num_thresh, b, dtype=torch.bool)

    for i, thresh in enumerate(px_thresh):
        i_est_E_param, i_success_mask = prepare_param_rel_pose(r_kp1, nn_r_i1_kp2, mutual_desc_matches_mask,
                                                               intrinsics1, intrinsics2, thresh)

        R_param_err[i] = (i_est_E_param[:, :3] - gt_E_param[:, :3]).norm(dim=-1)
        t_param_err[i] = (i_est_E_param[:, 3:] - gt_E_param[:, 3:]).norm(dim=-1)

        success_mask[i] = i_success_mask

    return R_param_err, t_param_err, success_mask


def pose_mAP(pose_err, pose_thresh, max_angle=180):
    angles = np.linspace(1, max_angle, num=max_angle)
    precision = [np.sum(pose_err < a) / len(pose_err) for a in angles]

    mAP = {thresh: np.mean(precision[:thresh]) for thresh in pose_thresh}

    return mAP, precision


# Legacy code

# def estimate_rel_poses_opencv(kp1, kp2, kp1_desc, kp2_desc,
#                               intrinsics1, intrinsics2, shift_scale1, shift_scale2,
#                               px_thresh, dd_measure,
#                               detailed=False):
#     i_intrinsics1 = intrinsics1.inverse()
#     i_intrinsics2 = intrinsics2.inverse()
#
#     m_desc_matches_mask, nn_desc_ids1 = get_mutual_desc_matches(kp1_desc, kp2_desc, dd_measure, 0.9)
#
#     o_kp1 = revert_data_transform(kp1, shift_scale1)
#     o_kp2 = revert_data_transform(kp2, shift_scale2)
#     nn_o_kp2 = select_kp(o_kp2, nn_desc_ids1)
#
#     est_rel_pose = torch.zeros(len(px_thresh), kp1.shape[0], 3, 4).to(kp1.device)
#
#     for i, p_th in enumerate(px_thresh):
#         for b in range(kp1.shape[0]):
#             b_gt_matches_mask = m_desc_matches_mask[b]
#
#             if b_gt_matches_mask.sum() < 8:
#                 continue
#
#             cv_kp1 = o_kp1[b][b_gt_matches_mask].cpu().numpy()
#             cv_nn_kp2 = nn_o_kp2[b][b_gt_matches_mask].cpu().numpy()
#
#             cv_intrinsics1 = intrinsics1[b].cpu().numpy()
#             cv_intrinsics2 = intrinsics2[b].cpu().numpy()
#
#             cv_i_intrinsics1 = i_intrinsics1[b].cpu().numpy()
#             cv_i_intrinsics2 = i_intrinsics2[b].cpu().numpy()
#
#             E_est_init = estimate_ess_mat_opencv(cv_kp1, cv_nn_kp2, cv_intrinsics1, cv_intrinsics2)
#             opt_res = least_squares(loss_fun, E_est_init, jac=loss_fun_jac,
#                                     args=(cv_kp1, cv_nn_kp2, cv_i_intrinsics1, cv_i_intrinsics2), method='lm')
#
#             if opt_res.success:
#                 R, _ = cv2.Rodrigues(opt_res.x[:3].reshape(-1))
#
#                 ab = np.append(opt_res.x[3:], 0)
#                 R_z, _ = cv2.Rodrigues(ab.reshape(-1))
#
#                 z_0 = np.array([0, 0, 1])
#                 t = R_z @ z_0
#
#                 est_rel_pose[i][b][:3,:3] = torch.tensor(R).to(kp1.device)
#                 est_rel_pose[i][b][:3, 3] = normalize(torch.tensor(-t).to(kp1.device), dim=-1)
#
#     if detailed:
#         return est_rel_pose, m_desc_matches_mask, nn_desc_ids1
#     else:
#         return est_rel_pose



# def estimate_rel_poses_gt_opengv(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                                  intrinsics1, intrinsics2, shift_scale1, shift_scale2,
#                                  px_thresh,
#                                  detailed=False):
#     gt_matches_mask, nn_kp_ids1 = \
#         get_best_gt_matches_old(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)
#
#     o_kp1 = revert_data_transform(kp1, shift_scale1)
#     o_kp2 = revert_data_transform(kp2, shift_scale2)
#     nn_o_kp2 = select_kp(o_kp2, nn_kp_ids1)
#
#     est_rel_pose = torch.zeros(len(px_thresh), kp1.shape[0], 3, 4).to(kp1.device)
#
#     if detailed:
#         est_inl_mask = torch.zeros(len(px_thresh), *kp1.shape[:2], dtype=torch.bool).to(kp1.device)
#
#     for i, p_th in enumerate(px_thresh):
#         i_gt_matches_mask = gt_matches_mask[i]
#
#         for b in range(kp1.shape[0]):
#             b_gt_matches_mask = i_gt_matches_mask[b]
#
#             if b_gt_matches_mask.sum() < 8:
#                 continue
#
#             cv_kp1 = o_kp1[b][b_gt_matches_mask].cpu().numpy()
#             nn_cv_kp2 = nn_o_kp2[b][b_gt_matches_mask].cpu().numpy()
#
#             T, b_inliers = \
#                 relative_pose_opengv(cv_kp1, nn_cv_kp2, intrinsics1[b].cpu().numpy(),
#                                      intrinsics2[b].cpu().numpy(), p_th)
#
#             est_rel_pose[i][b] = torch.tensor(T).to(kp1.device)
#             est_rel_pose[i][b][:3, 3] = normalize(est_rel_pose[i][b][:3, 3], dim=-1)
#
#             if detailed:
#                 est_inl_mask[i][b][b_gt_matches_mask] = torch.tensor(b_inliers).to(kp1.device)
#
#     if detailed:
#         return est_rel_pose, est_inl_mask, gt_matches_mask, nn_kp_ids1
#     else:
#         return est_rel_pose
#
#
# def estimate_rel_poses_sift_opengv(sift_kp1, sift_kp2, intrinsics1, intrinsics2, px_thresh):
#     est_rel_pose = torch.zeros(len(px_thresh), 3, 4)
#     est_inl_mask = torch.zeros(len(px_thresh), sift_kp1.shape[0], dtype=torch.bool)
#
#     for i, p_th in enumerate(px_thresh):
#         T, b_inliers = \
#             relative_pose_opengv(sift_kp1, sift_kp2, intrinsics1, intrinsics2, p_th)
#
#         est_rel_pose[i] = torch.tensor(T)
#         est_rel_pose[i][:3, 3] = normalize(est_rel_pose[i][:3, 3], dim=-1)
#
#         est_inl_mask[i] = torch.tensor(b_inliers)
#
#     return est_rel_pose, est_inl_mask


# def nn_mAP(w_kp1, kp2, wv_kp1_mask, wv_kp2_mask, pixel_thresholds, kp1_desc, kp2_desc, desc_dist_measure,
#            detailed=False):
#     """
#     :param w_kp1: B x N x 2; keypoints on the first image projected to the second
#     :param kp2: B x N x 2; keypoints on the second image
#     :param wv_kp1_mask: B x N; keypoints on the first image which are visible on the second
#     :param wv_kp2_mask: B x N; keypoints on the second image which are visible on the first
#     :param pixel_thresholds: P :type torch.float
#     :param kp1_desc: B x N x C; descriptors for keypoints on the first image
#     :param kp2_desc: B x N x C; descriptors for keypoints on the second image
#     :param desc_dist_measure: measure of descriptor distance. Can be L2-norm or similarity measure
#     :param detailed: return detailed information :type bool
#     """
#     b, n = wv_kp1_mask.shape
#
#     # Calculate pairwise distance/similarity measure
#     if desc_dist_measure is DescriptorDistance.INV_COS_SIM:
#         desc_sim = smooth_inv_cos_sim_mat(kp1_desc, kp2_desc)
#     else:
#         desc_sim = calculate_distance_matrix(kp1_desc, kp2_desc)
#
#     nn_desc_values, nn_desc_ids = desc_sim.min(dim=-1)
#
#     # Remove duplicate matches in each scene
#     unique_match_mask = calculate_unique_match_mask(nn_desc_values, nn_desc_ids)
#
#     # Calculate pairwise keypoints distances
#     kp_dist = calculate_distance_matrix(w_kp1, kp2)
#     kp_dist = mask_non_visible_pairs(kp_dist, wv_kp1_mask, wv_kp2_mask)
#
#     # Retrieve correspondent keypoints
#     nn_kp_values = torch.gather(kp_dist, -1, nn_desc_ids.view(b, n, 1)).view(b, n)
#
#     if detailed:
#         nn_mAP_scores = torch.zeros(pixel_thresholds.shape[0], w_kp1.shape[0])
#         precisions = torch.zeros(pixel_thresholds.shape[0], w_kp1.shape[0], w_kp1.shape[1])
#         recalls = torch.zeros(pixel_thresholds.shape[0], w_kp1.shape[0], w_kp1.shape[1])
#     else:
#         nn_mAP_scores = torch.zeros_like(pixel_thresholds)
#
#     for i, thresh in enumerate(pixel_thresholds):
#         # Threshold correspondences
#         t_match_mask = nn_kp_values.le(thresh) * unique_match_mask
#
#         # Calculate number of matches for each scene
#         t_matches = t_match_mask.sum(dim=-1).float()
#
#         # Calculate tp and fp
#         tp = torch.cumsum(t_match_mask == True, dim=-1).float()
#         fp = torch.cumsum((t_match_mask == False) * wv_kp1_mask, dim=-1).float()
#
#         precision = tp / (tp + fp).clamp(min=1e-8)
#         recall = tp / t_matches.view(-1, 1).clamp(min=1e-8)
#
#         if detailed:
#             nn_mAP_scores[i] = torch.trapz(precision, recall)
#             precisions[i] = precision.sort(dim=-1)[0]
#             recalls[i] = recall.sort(dim=-1)[0]
#         else:
#             nn_mAP_scores[i] = torch.trapz(precision, recall).mean()
#
#     if detailed:
#         return nn_mAP_scores, precisions, recalls
#     else:
#         return nn_mAP_scores

# def epipolar_distance_score(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, intrinsics1, intrinsics2, extrinsics1, extrinsics2, shift_scale1, shift_scale2):
#     gt_matches_mask, nn_kp_ids1 = get_best_gt_matches_old(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, [5.0], mutual=False)
#
#     o_kp1 = revert_data_transform(kp1, shift_scale1)
#     o_kp2 = revert_data_transform(kp2, shift_scale2)
#     nn_o_kp2 = select_kp(o_kp2, nn_kp_ids1)
#
#     F = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2)
#
#     ep_dist = epipolar_distance(o_kp1, nn_o_kp2, F)
#
#     score = (ep_dist * gt_matches_mask[0].float()).sum(dim=-1) / gt_matches_mask[0].float().sum(dim=-1).clamp(min=1e-8)
#     return score.mean()
