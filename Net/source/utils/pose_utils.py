import cv2
import numpy as np
from scipy.optimize import least_squares

from torch.nn.functional import normalize

import torch

from torch.autograd import Function

from opensfm import multiview
from opensfm.types import BrownPerspectiveCamera
from opensfm.matching import _compute_inliers_bearings

from Net.source.utils.matching_utils import get_mutual_desc_matches, select_kp, get_gt_matches

from Net.source.utils.math_utils import revert_data_transform, change_intrinsics, parametrize_pose, \
    compose_gt_transform, epipolar_distance


def estimate_rel_pose_opengv(kp1, kp2, kp1_desc, kp2_desc, shift_scale1, shift_scale2, intrinsics1, intrinsics2,
                             px_thresh, dd_measure, detailed=False):
    mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, dd_measure, 0.9)

    r_kp1 = revert_data_transform(kp1, shift_scale1)
    r_kp2 = revert_data_transform(kp2, shift_scale2)

    nn_r_kp2 = select_kp(r_kp2, nn_desc_ids)

    rel_pose = prepare_rel_pose(r_kp1, nn_r_kp2, mutual_desc_matches_mask, intrinsics1, intrinsics2, px_thresh, detailed)

    if detailed:
        return rel_pose[0], rel_pose[1], mutual_desc_matches_mask, nn_desc_ids
    else:
        return rel_pose[0]


def estimate_param_rel_pose_opencv(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, shift_scale1, shift_scale2,
                                   intrinsics1, intrinsics2, px_thresh, detailed=False):
    mutual_gt_matches_mask, nn_kp_ids = get_gt_matches(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)

    r_kp1 = revert_data_transform(kp1, shift_scale1)
    r_kp2 = revert_data_transform(kp2, shift_scale2)

    nn_r_kp2 = select_kp(r_kp2, nn_kp_ids)
    nn_r_i1_kp2 = change_intrinsics(nn_r_kp2, intrinsics2, intrinsics1)

    est_E_param, success_mask = prepare_param_rel_pose(r_kp1, nn_r_i1_kp2, mutual_gt_matches_mask, intrinsics1, intrinsics2)

    if detailed:
        return est_E_param, success_mask, mutual_gt_matches_mask, nn_kp_ids
    else:
        return est_E_param, success_mask


class ParamRelPose(Function):

    @staticmethod
    def forward(ctx, r_kp1, nn_r_i1_kp2, mutual_gt_matches_mask, intrinsics1, intrinsics2):
        est_E_param, success_mask = prepare_param_rel_pose(r_kp1, nn_r_i1_kp2, mutual_gt_matches_mask,
                                                           intrinsics1, intrinsics2)

        ctx.save_for_backward(est_E_param, success_mask, r_kp1, nn_r_i1_kp2, mutual_gt_matches_mask,
                              intrinsics1, intrinsics2)
        ctx.mark_non_differentiable(success_mask)

        return est_E_param, success_mask

    @staticmethod
    def backward(ctx, d_est_E_param, d_success_mask):
        est_E_param, success_mask, r_kp1, nn_r_i1_kp2, mutual_gt_matches_mask, intrinsics1, intrinsics2 = ctx.saved_tensors

        d_r_kp1, d_nn_r_kp2 = prepare_param_rel_pose_grad(d_est_E_param, est_E_param, success_mask, r_kp1, nn_r_i1_kp2,
                                                          mutual_gt_matches_mask, intrinsics1, intrinsics2)

        return d_r_kp1, d_nn_r_kp2, \
               torch.zeros_like(mutual_gt_matches_mask).to(mutual_gt_matches_mask.device), \
               torch.zeros_like(intrinsics1).to(intrinsics1.device), \
               torch.zeros_like(intrinsics2).to(intrinsics2.device)


class ParamTruncRelPose(Function):

    @staticmethod
    def forward(ctx, r_kp1, nn_r_i1_kp2, nn_r_kp2, mutual_gt_matches_mask, intrinsics1, intrinsics2, extrinsics1, extrinsics2):
        est_E_param, success_mask = prepare_param_rel_pose(r_kp1, nn_r_i1_kp2, mutual_gt_matches_mask,
                                                           intrinsics1, intrinsics2)

        ctx.save_for_backward(est_E_param, success_mask, r_kp1, nn_r_i1_kp2, nn_r_kp2, mutual_gt_matches_mask,
                              intrinsics1, intrinsics2, extrinsics1, extrinsics2)
        ctx.mark_non_differentiable(success_mask)

        return est_E_param, success_mask


    @staticmethod
    def backward(ctx, d_est_E_param, d_success_mask):
        est_E_param, success_mask, r_kp1, nn_r_i1_kp2, nn_r_kp2, mutual_gt_matches_mask, \
        intrinsics1, intrinsics2, extrinsics1, extrinsics2 = ctx.saved_tensors

        d_r_kp1, d_nn_r_kp2 = prepare_param_rel_pose_grad(d_est_E_param, est_E_param, success_mask, r_kp1, nn_r_i1_kp2,
                                                          mutual_gt_matches_mask, intrinsics1, intrinsics2)

        F = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2)

        init_ep_dist1 = epipolar_distance(nn_r_kp2, r_kp1, F.transpose(1, 2))
        grad_ep_dist1 = epipolar_distance(nn_r_kp2, r_kp1 - d_r_kp1, F.transpose(1, 2))

        coll_grad_mask = (grad_ep_dist1 - init_ep_dist1) < 0

        d_r_kp1 = d_r_kp1 * coll_grad_mask.unsqueeze(-1).float()
        d_nn_r_kp2 = d_nn_r_kp2 * coll_grad_mask.unsqueeze(-1).float()

        return d_r_kp1, d_nn_r_kp2, \
               torch.zeros_like(nn_r_kp2).to(nn_r_kp2.device), \
               torch.zeros_like(mutual_gt_matches_mask).to(mutual_gt_matches_mask.device), \
               torch.zeros_like(intrinsics1).to(intrinsics1.device), \
               torch.zeros_like(intrinsics2).to(intrinsics2.device), \
               torch.zeros_like(extrinsics1).to(extrinsics1.device), \
               torch.zeros_like(extrinsics2).to(extrinsics2.device)


def relative_pose_opengv(r_kp1, nn_r_kp2, intrinsics1, intrinsics2, px_thresh):
    camera1 = intrinsics2camera(intrinsics1)
    camera2 = intrinsics2camera(intrinsics2)

    bearing_vectors1 = camera1.pixel_bearing_many(r_kp1)
    bearing_vectors2 = camera2.pixel_bearing_many(nn_r_kp2)

    # Convert pixel threshold to angular
    avg_focal_length = (camera1.focal_x + camera1.focal_y + camera2.focal_x + camera2.focal_y) / 4
    angle_thresh = np.arctan2(px_thresh, avg_focal_length)

    T = multiview.relative_pose_ransac(bearing_vectors1, bearing_vectors2, b"STEWENIUS",
                                       1 - np.cos(angle_thresh), 5000, 0.99999)
    inliers = _compute_inliers_bearings(bearing_vectors1, bearing_vectors2, T, angle_thresh)

    T = multiview.relative_pose_optimize_nonlinear(bearing_vectors1[inliers], bearing_vectors2[inliers],
                                                   T[:3, 3], T[:3, :3])
    inliers = _compute_inliers_bearings(bearing_vectors1, bearing_vectors2, T, angle_thresh)

    return T, inliers


def relative_param_pose_opencv(r_kp1, nn_r_kp2, intrinsics1):
    # TODO. Do we need thresholding here?
    est_E, _ = cv2.findEssentialMat(r_kp1, nn_r_kp2, intrinsics1, method=cv2.RANSAC)
    _, est_R, est_t, _ = cv2.recoverPose(est_E, r_kp1, nn_r_kp2, intrinsics1)

    est_t = est_t.reshape(-1)
    est_t = est_t / np.linalg.norm(est_t)

    est_E_param = parametrize_pose(est_R, est_t)

    return est_E_param


def prepare_rel_pose(r_kp1, nn_r_kp2, mutual_desc_matches_mask, intrinsics1, intrinsics2, px_thresh, detailed):
    num_thresh = len(px_thresh)

    est_rel_pose = torch.zeros(num_thresh, r_kp1.shape[0], 3, 4).to(r_kp1.device)

    if detailed:
        est_inl_mask = torch.zeros(num_thresh, *r_kp1.shape[:2], dtype=torch.bool).to(r_kp1.device)

    for i, thresh in enumerate(px_thresh):
        for j in range(r_kp1.shape[0]):
            b_mutual_desc_matches_mask = mutual_desc_matches_mask[j]

            if b_mutual_desc_matches_mask.sum() < 8:
                continue

            cv_kp1 = r_kp1[j][b_mutual_desc_matches_mask].cpu().detach().numpy()
            nn_cv_kp2 = nn_r_kp2[j][b_mutual_desc_matches_mask].cpu().detach().numpy()

            T, b_inliers = relative_pose_opengv(cv_kp1, nn_cv_kp2,
                                                intrinsics1[j].cpu().numpy(), intrinsics2[j].cpu().numpy(), thresh)

            est_rel_pose[i][j] = torch.tensor(T).to(r_kp1.device)
            est_rel_pose[i][j][:3, 3] = normalize(est_rel_pose[i][j][:3, 3], dim=-1)

            if detailed:
                est_inl_mask[i][j][b_mutual_desc_matches_mask] = torch.tensor(b_inliers).to(r_kp1.device)

    if detailed:
        return est_rel_pose, est_inl_mask
    else:
        return est_rel_pose


def prepare_param_rel_pose(r_kp1, nn_r_kp2, mutual_gt_matches_mask, intrinsics1, intrinsics2):
    b = r_kp1.shape[0]
    est_E_param = torch.zeros(b, 5).to(r_kp1.device)

    success_mask = torch.zeros(b, dtype=torch.bool).to(r_kp1.device)

    for i in range(b):
        b_mutual_gt_matches_mask = mutual_gt_matches_mask[i]

        if b_mutual_gt_matches_mask.sum() < 8:
            continue

        cv_kp1 = r_kp1[i][b_mutual_gt_matches_mask].cpu().detach().numpy()
        cv_nn_kp2 = nn_r_kp2[i][b_mutual_gt_matches_mask].cpu().detach().numpy()

        cv_intrinsics1 = intrinsics1[i].cpu().numpy()

        cv_i_intrinsics1 = intrinsics1[i].inverse().cpu().numpy()
        cv_i_intrinsics2 = intrinsics2[i].inverse().cpu().numpy()

        est_init_E_param = relative_param_pose_opencv(cv_kp1, cv_nn_kp2, cv_intrinsics1)

        opt_res = least_squares(loss_fun, est_init_E_param, jac=loss_fun_jac,
                                args=(cv_kp1, cv_nn_kp2, cv_i_intrinsics1, cv_i_intrinsics2), method='lm')

        if opt_res.success:
            est_E_param[i] = torch.tensor(opt_res.x).to(r_kp1.device)

            success_mask[i] = True

    return est_E_param, success_mask


def prepare_param_rel_pose_grad(d_est_E_param, est_E_param, success_mask, r_kp1, nn_r_kp2, mutual_gt_matches_mask,
                                intrinsics1, intrinsics2):
    b = r_kp1.shape[0]
    d_r_kp1 = torch.zeros_like(r_kp1).to(r_kp1.device)
    d_nn_r_kp2 = torch.zeros_like(nn_r_kp2).to(nn_r_kp2.device)

    for i in range(b):
        if success_mask[i]:
            cv_est_E_param = est_E_param[i].cpu().numpy()

            b_mutual_gt_matches_mask = mutual_gt_matches_mask[i]

            cv_kp1 = r_kp1[i][b_mutual_gt_matches_mask].cpu().numpy()
            cv_nn_kp2 = nn_r_kp2[i][b_mutual_gt_matches_mask].cpu().numpy()

            cv_i_intrinsics1 = intrinsics1[i].inverse().cpu().numpy()
            cv_i_intrinsics2 = intrinsics2[i].inverse().cpu().numpy()

            d_pose_r_kp1, d_pose_nn_r_kp2 = compute_pose_pt_derivative(cv_est_E_param, cv_i_intrinsics1,
                                                                       cv_i_intrinsics2,
                                                                       cv_kp1, cv_nn_kp2)

            if d_pose_r_kp1 is not None:
                d_pose_r_kp1 = torch.tensor(d_pose_r_kp1, dtype=torch.float).to(d_r_kp1.device)
                d_pose_nn_r_kp2 = torch.tensor(d_pose_nn_r_kp2, dtype=torch.float).to(d_nn_r_kp2.device)

                # TODO. Crude singularity filtering. REDO
                m1 = d_pose_r_kp1.view(-1, 10).norm(dim=-1).abs() > 0.1
                m2 = d_pose_nn_r_kp2.view(-1, 10).norm(dim=-1).abs() > 0.1
                m = m1.sum() + m2.sum()

                if m == 0:
                    i_d_r_kp1 = (d_est_E_param[i].unsqueeze(0).unsqueeze(0) @ d_pose_r_kp1).squeeze(1)
                    i_d_nn_r_kp2 = (d_est_E_param[i].unsqueeze(0).unsqueeze(0) @ d_pose_nn_r_kp2).squeeze(1)

                    d_r_kp1[i][b_mutual_gt_matches_mask] = i_d_r_kp1
                    d_nn_r_kp2[i][b_mutual_gt_matches_mask] = i_d_nn_r_kp2

    return d_r_kp1, d_nn_r_kp2


"""
Support utils
"""


def intrinsics2camera(intrinsics):
    """
    :param intrinsics: 3 x 3
    """
    camera = BrownPerspectiveCamera()

    camera.focal_x = intrinsics[0, 0]
    camera.focal_y = intrinsics[1, 1]

    camera.c_x = intrinsics[0, 2]
    camera.c_y = intrinsics[1, 2]

    camera.k1 = 0
    camera.k2 = 0
    camera.k3 = 0
    camera.p1 = 0
    camera.p2 = 0

    return camera


def loss_fun(x, kp1, kp2, iK1, iK2):
    E = decode_essmat(x)
    F = iK2.T @ E @ iK1
    errs, errs_rev = compute_epipolar_errors_opt(F, kp1, kp2)
    return np.concatenate([errs, errs_rev], axis=0)


def loss_fun_jac(x, kp1, kp2, iK1, iK2):
    E, dE = decode_essmat(x, True)
    F = iK2.T @ E @ iK1
    dF = np.matmul(iK2.T.reshape(1, 3, 3), np.matmul(dE, iK1.reshape(1, 3, 3)))
    _, _, derrs, derrs_rev = compute_epipolar_errors_opt(F, kp1, kp2, dF)
    dres = np.concatenate([derrs.T, derrs_rev.T], axis=0)
    return dres


def decode_essmat(x, is_jac=False):
    # assuming jac is 3 x 9
    R, jac = cv2.Rodrigues(x[0:3])
    x2 = np.zeros(3)
    x2[0:2] = x[3:5]
    R2, jac2 = cv2.Rodrigues(x2)
    t = R2[:, 2]
    cp_t = vec_to_cross(t)
    E = cp_t @ R
    if not is_jac:
        return E

    jac_big = np.zeros((5, 3, 3))
    jac_big[0:3, :, :] = np.matmul(cp_t.reshape(1, 3, 3), jac.reshape(3, 3, 3))

    dcpt1 = np.zeros((3, 3))
    dcpt1[1, 2] = -1
    dcpt1[2, 1] = 1
    jac_big[3:5, :, :] = (dcpt1 @ R).reshape(1, 3, 3) * jac2[0:2, 2].reshape(2, 1, 1)

    dcpt2 = np.zeros((3, 3))
    dcpt2[0, 2] = 1
    dcpt2[2, 0] = -1
    jac_big[3:5, :, :] += (dcpt2 @ R).reshape(1, 3, 3) * jac2[0:2, 5].reshape(2, 1, 1)

    dcpt3 = np.zeros((3, 3))
    dcpt3[0, 1] = -1
    dcpt3[1, 0] = 1
    jac_big[3:5, :, :] += (dcpt3 @ R).reshape(1, 3, 3) * jac2[0:2, 8].reshape(2, 1, 1)

    return E, jac_big


def compute_epipolar_errors_opt(F, kp1, kp2, dF=None, is_pt_diff=False):
    n = kp1.shape[0]
    kp1h = np.concatenate([kp1, np.ones((n, 1))], axis=1)
    kp2h = np.concatenate([kp2, np.ones((n, 1))], axis=1)
    luh = kp1h @ F.T
    lambdas = np.linalg.norm(luh[:, 0:2], axis=1).reshape(n, 1)
    lh = luh / lambdas
    errs = np.matmul(lh.reshape(n, 1, 3), kp2h.reshape(n, 3, 1)).reshape(-1)

    luh_rev = kp2h @ F
    lambdas_rev = np.linalg.norm(luh_rev[:, 0:2], axis=1).reshape(n, 1)
    lh_rev = luh_rev / lambdas_rev
    errs_rev = np.matmul(lh_rev.reshape(n, 1, 3), kp1h.reshape(n, 3, 1)).reshape(-1)
    if dF is None:
        return errs, errs_rev

    if not is_pt_diff:
        dFt = dF.transpose(0, 2, 1)
        derrs = diff_ptl_errs(kp1h, kp2h, luh, lambdas, dFt)
        derrs_rev = diff_ptl_errs(kp2h, kp1h, luh_rev, lambdas_rev, dF)
        return errs, errs_rev, derrs, derrs_rev

    dFt = dF.transpose(0, 2, 1)
    derrs, derr_pt = diff_ptl_errs(kp1h, kp2h, luh, lambdas, dFt, F=F, lh=lh)
    Ft = F.T
    derrs_rev, derr_rev_pt = diff_ptl_errs(kp2h, kp1h, luh_rev, lambdas_rev, dF, F=Ft, lh=lh_rev)

    return errs, errs_rev, derrs, derrs_rev, derr_pt, derr_rev_pt


def diff_ptl_errs(kp1h, kp2h, luh, lambdas, dFt, F=None, lh=None):
    n = kp1h.shape[0]
    dluh = np.matmul(kp1h.reshape(1, n, 3), dFt)  # 5 x n x 3
    dlnh = (1.0 / lambdas).reshape(n, 1, 1) * np.eye(3).reshape(1, 3, 3)  # n, 3, 3
    lzuh = np.concatenate([luh[:, 0:2], np.zeros((n, 1))], axis=1)
    dlnh -= (lambdas ** (-3)).reshape(n, 1, 1) * np.matmul(luh.reshape(n, 3, 1), lzuh.reshape(n, 1, 3))
    dlh = np.matmul(dlnh.reshape(1, n, 3, 3), dluh.reshape(5, n, 3, 1)).reshape(5, n, 3)
    derrs = np.matmul(dlh.reshape(5, n, 1, 3), kp2h.reshape(1, n, 3, 1)).reshape(5, n)
    if F is None:
        return derrs
    derr_pt = diff_ptl_errs_pts(kp1h, kp2h, lh, dlh, dlnh, F, dFt, lambdas, luh, lzuh, dluh)
    return derrs, derr_pt


def diff_ptl_errs_pts(kp1h, kp2h, lh, dlh, dlnh, F, dFt, lambdas, luh, lzuh, dluh):
    n = kp1h.shape[0]

    derr2 = lh.transpose()[0:2, :]  # 3 x n
    d2err2 = dlh.reshape(5, n, 3).transpose(0, 2, 1)[:, 0:2, :]  # 5 x 3 x n

    derr = np.matmul(kp2h.reshape(n, 1, 3), np.matmul(dlnh.reshape(n, 3, 3),
                                                      F.reshape(1, 3, 3))).reshape(n, 3)[:, 0:2].T
    # d2err = np.matmul(kp2h.reshape(1, n, 1, 3),
    #                   np.matmul(dlnh.reshape(1, n, 3, 3),
    #                   dFt.reshape(5, 1, 3, 3).transpose(0, 1, 3, 2))).reshape(5, n, 3)[:, :, 0:2].transpose(0, 2, 1)
    luh_dx = F.transpose().reshape(1, 3, 3)
    dlambda = np.matmul(luh_dx, lzuh.reshape(n, 3, 1))  # n x 3 x 1
    d = np.eye(3).reshape(1, 1, 3, 3) * dlambda.reshape(n, 3, 1, 1) * (
        -lambdas.reshape(n, 1, 1, 1) ** (-3))  # n x 3 x 3 x 3
    d += 3 * lambdas.reshape(n, 1, 1, 1) ** (-5) * np.matmul(luh.reshape(n, 1, 3, 1),
                                                             lzuh.reshape(n, 1, 1, 3)) * dlambda.reshape(n, 3, 1, 1)

    d += -(lambdas ** (-3)).reshape(n, 1, 1, 1) * np.matmul(luh_dx.reshape(1, 3, 3, 1), lzuh.reshape(n, 1, 1, 3))
    F0 = np.copy(F)
    F0[2, :] = 0
    luh0_dx = F0.transpose().reshape(1, 3, 1, 3)
    d += -(lambdas ** (-3)).reshape(n, 1, 1, 1) * np.matmul(luh.reshape(n, 1, 3, 1), luh0_dx)

    # dluh: 5 x n x 3
    dmult = np.matmul(d.reshape(1, n, 3, 3, 3), dluh.reshape(5, n, 1, 3, 1)).reshape(5, n, 3, 3, 1)
    dmult += np.matmul(dlnh.reshape(1, n, 1, 3, 3), dFt.reshape(5, 1, 3, 3, 1)).reshape(5, n, 3, 3, 1)  # old

    d2err = np.matmul(kp2h.reshape(1, n, 1, 1, 3), dmult).reshape(5, n, 3)[:, :, 0:2].transpose(0, 2, 1)

    return derr, d2err, derr2, d2err2


def compute_pose_pt_derivative(x, iK1, iK2, kp1, kp2):
    E, dE = decode_essmat(x, True)

    # All condition numbers are defined
    if np.sum(np.isinf(np.linalg.cond(dE))) == 0:
        F = iK2.T @ E @ iK1
        dF = np.matmul(iK2.T.reshape(1, 3, 3), np.matmul(dE, iK1.reshape(1, 3, 3)))
        errs, errs_rev, derrs, derrs_rev, \
        derr_pt, derr_pt_rev = compute_epipolar_errors_opt(F, kp1, kp2, dF=dF, is_pt_diff=True)
        # fwd err
        D, D2 = dpose_derivative(errs, derrs, derr_pt, errs_rev, derrs_rev, derr_pt_rev)
        return D, D2
    else:
        return None, None


def dpose_derivative(errs, derrs, derr_pt, errs_rev, derrs_rev, derr_pt_rev):
    JJ = np.zeros((5, 5))
    n = errs.shape[0]
    for i in range(0, n):
        JJ += np.matmul(derrs[:, i].reshape(-1, 1), derrs[:, i].reshape(1, -1))
        JJ += np.matmul(derrs_rev[:, i].reshape(-1, 1), derrs_rev[:, i].reshape(1, -1))

    # TODO. Inversion may lead to instability use LU-decomposition

    JJi = np.linalg.inv(JJ)
    D = np.zeros((n, 5, 2))
    D2 = np.zeros((n, 5, 2))

    p_derr, p_d2err, p_derr2, p_d2err2 = derr_pt
    p_derr_rev, p_d2err_rev, p_derr2_rev, p_d2err2_rev = derr_pt_rev

    for i in range(0, n):
        rhs = -(np.matmul(derrs[:, i].reshape(-1, 1), p_derr[:, i].reshape(1, -1)) + p_d2err[:, :, i] * errs[
            i])  # 3 x n
        rhs += -(np.matmul(derrs_rev[:, i].reshape(-1, 1), p_derr2_rev[:, i].reshape(1, -1)) + p_d2err2_rev[:, :, i] *
                 errs_rev[i])  # 3 x n
        D[i] = JJi @ rhs

        rhs2 = -(np.matmul(derrs[:, i].reshape(-1, 1), p_derr2[:, i].reshape(1, -1)) + p_d2err2[:, :, i] * errs[
            i])  # 3 x n
        rhs2 += -(np.matmul(derrs_rev[:, i].reshape(-1, 1), p_derr_rev[:, i].reshape(1, -1)) + p_d2err_rev[:, :, i] *
                  errs_rev[i])  # 3 x n
        D2[i] = JJi @ rhs2

    return D, D2


def vec_to_cross(x):
    return np.asarray([[0, -x[2], x[1]],
                       [x[2], 0, -x[0]],
                       [-x[1], x[0], 0]])


# Legacy code

def test_pose_pt_derivative(x, iK1, iK2, kp1, kp2):
    D, D2 = compute_pose_pt_derivative(x, iK1, iK2, kp1, kp2)

    n = kp1.shape[0]
    delta = 1e-1
    for i in range(0, n):
        for j in range(0, 2):
            kp1p = np.copy(kp1)
            kp1p[i, j] += delta
            opt_res = least_squares(loss_fun, x, jac=loss_fun_jac, args=(kp1p, kp2, iK1, iK2), method='lm')
            if not opt_res.success:
                print('fail opt')
            xp = opt_res.x
            dx_fd = 1.0 / delta * (xp - x)
            dx = D[i, :, j]
            print('{},{}: {}'.format(i, j, np.linalg.norm(dx - dx_fd)))
            # print(dx_fd)
            # print('-')
    print('-')
    for i in range(0, n):
        for j in range(0, 2):
            kp2p = np.copy(kp2)
            kp2p[i, j] += delta
            opt_res = least_squares(loss_fun, x, jac=loss_fun_jac, args=(kp1, kp2p, iK1, iK2), method='lm')
            if not opt_res.success:
                print('fail opt')
            xp = opt_res.x
            dx_fd = 1.0 / delta * (xp - x)
            dx = D2[i, :, j]
            print('{},{}: {}'.format(i, j, np.linalg.norm(dx - dx_fd)))
            print(dx)
            # print(dx_fd)
            # print('-')
    print('-')

# class EstimateParamEssMat2(Function):
#
#     @staticmethod
#     def forward(ctx, o_kp1, nn_o_kp2, intrinsics1, intrinsics2, match_mask, F):
#         i_intrinsics1 = intrinsics1.inverse()
#         i_intrinsics2 = intrinsics2.inverse()
#
#         b = o_kp1.shape[0]
#         E_ests = torch.zeros(b, 5).to(o_kp1.device)
#         E_init_ests = torch.zeros(b, 5).to(o_kp1.device)
#
#         success_mask = torch.zeros(b, dtype=torch.bool).to(o_kp1.device)
#
#         for i in range(b):
#             cv_kp1 = o_kp1[i, match_mask[i]].cpu().numpy()
#             cv_nn_kp2 = nn_o_kp2[i, match_mask[i]].cpu().numpy()
#
#             if cv_kp1.shape[0] > 10:
#
#                 cv_intrinsics1 = intrinsics1[i].cpu().numpy()
#                 cv_intrinsics2 = intrinsics2[i].cpu().numpy()
#
#                 cv_i_intrinsics1 = i_intrinsics1[i].cpu().numpy()
#                 cv_i_intrinsics2 = i_intrinsics2[i].cpu().numpy()
#
#                 E_est_init = estimate_ess_mat_opencv(cv_kp1, cv_nn_kp2, cv_intrinsics1, cv_intrinsics2)
#
#                 opt_res = least_squares(loss_fun, E_est_init, jac=loss_fun_jac,
#                                         args=(cv_kp1, cv_nn_kp2, cv_i_intrinsics1, cv_i_intrinsics2), method='lm')
#
#                 if opt_res.success:
#                     E_init_ests[i] = torch.tensor(E_est_init).to(o_kp1.device)
#                     E_ests[i] = torch.tensor(opt_res.x).to(o_kp1.device)
#
#                     success_mask[i] = True
#
#         ctx.save_for_backward(E_ests, success_mask, o_kp1, nn_o_kp2, i_intrinsics1, i_intrinsics2, match_mask, F)
#         ctx.mark_non_differentiable(E_init_ests, success_mask)
#
#         return E_ests, E_init_ests, success_mask
#
#     @staticmethod
#     def backward(ctx, dE_est_outputs, dE_init_ests, d_success_mask):
#         E_ests, success_mask, o_kp1, nn_o_kp2, i_intrinsics1, i_intrinsics2, match_mask, F = ctx.saved_tensors
#
#         b = o_kp1.shape[0]
#         ds_o_kp1 = torch.zeros_like(o_kp1).to(o_kp1.device)
#         ds_nn_o_kp2 = torch.zeros_like(nn_o_kp2).to(nn_o_kp2.device)
#
#         line1 = to_homogeneous2(nn_o_kp2) @ F
#         norm_line1 = line1 / line1[..., :2].norm(dim=-1).clamp(min=1e-16).unsqueeze(-1)
#
#         for i in range(b):
#             if success_mask[i]:
#                 E_est = E_ests[i].detach().cpu().numpy()
#
#                 cv_kp1 = o_kp1[i, match_mask[i]].cpu().numpy()
#                 cv_nn_kp2 = nn_o_kp2[i, match_mask[i]].cpu().numpy()
#
#                 cv_i_intrinsics1 = i_intrinsics1[i].cpu().numpy()
#                 cv_i_intrinsics2 = i_intrinsics2[i].cpu().numpy()
#
#                 norm_of_line_i = norm_line1[i, match_mask[i]].cpu().detach().numpy()
#
#                 d_E_o_kp1, d_E_nn_o_kp2 = compute_pose_pt_derivative(E_est, cv_i_intrinsics1, cv_i_intrinsics2, cv_kp1,
#                                                                      cv_nn_kp2)
#
#                 if d_E_o_kp1 is not None:
#                     d_E_o_kp1 = torch.tensor(d_E_o_kp1, dtype=torch.float).to(o_kp1.device)
#                     d_E_nn_o_kp2 = torch.tensor(d_E_nn_o_kp2, dtype=torch.float).to(nn_o_kp2.device)
#
#                     m1 = d_E_o_kp1.view(-1, 10).norm(dim=-1).abs() > 0.1
#                     m2 = d_E_nn_o_kp2.view(-1, 10).norm(dim=-1).abs() > 0.1
#                     m = m1.sum() + m2.sum()
#
#                     if m == 0:
#                         d_o_kp1 = (dE_est_outputs[i].unsqueeze(0).unsqueeze(0) @ d_E_o_kp1).squeeze(1)
#                         d_nn_o_kp2 = (dE_est_outputs[i].unsqueeze(0).unsqueeze(0) @ d_E_nn_o_kp2).squeeze(1)
#
#                         begin = np.abs((to_homogeneous(cv_kp1) * norm_of_line_i).sum(axis=-1))
#                         end = np.abs(
#                             (to_homogeneous(cv_kp1 - d_o_kp1.cpu().detach().numpy()) * norm_of_line_i).sum(axis=-1))
#
#                         mask = torch.tensor((end - begin) < 0).to(F.device)
#
#                         # initial_dist = (to_homogeneous(cv_kp1) * norm_of_line_i).sum(axis=-1)
#                         # final_dist = (to_homogeneous(cv_kp1 + d_o_kp1.cpu().detach().numpy()) * norm_of_line_i).sum(axis=-1)
#
#                         # mask = (final_dist - initial_dist) > 0
#                         # mask = torch.tensor(mask).to(F.device)
#
#                         # print(normalize(d_o_kp1))
#
#                         ds_o_kp1[i, match_mask[i]] = d_o_kp1 * mask.float().unsqueeze(-1)
#                         ds_nn_o_kp2[i, match_mask[i]] = d_nn_o_kp2 * mask.float().unsqueeze(-1)
#
#                         # ds_o_kp1[i, match_mask[i]] = d_o_kp1
#                         # ds_nn_o_kp2[i, match_mask[i]] = d_nn_o_kp2
#
#         return ds_o_kp1, ds_nn_o_kp2, \
#                torch.zeros_like(i_intrinsics1).to(i_intrinsics1.device), \
#                torch.zeros_like(i_intrinsics2).to(i_intrinsics2.device), \
#                torch.zeros_like(match_mask).to(match_mask.device), \
#                torch.zeros_like(F).to(F.device)

#
# def estimate_pose_grad(o_kp1, nn_o_kp2, intrinsics1, intrinsics2, match_mask):
#     i_intrinsics1 = intrinsics1.inverse()
#     i_intrinsics2 = intrinsics2.inverse()
#
#     b = o_kp1.shape[0]
#     n = o_kp1.shape[1]
#     success_mask = torch.zeros(b, dtype=torch.bool).to(o_kp1.device)
#
#     ds_o_kp1 = torch.zeros(b, n).to(o_kp1.device)
#     ds_nn_o_kp2 = torch.zeros(b, n).to(nn_o_kp2.device)
#
#     for i in range(b):
#         cv_kp1 = o_kp1[i, match_mask[i]].cpu().numpy()
#         cv_nn_kp2 = nn_o_kp2[i, match_mask[i]].cpu().numpy()
#
#         if cv_kp1.shape[0] > 10:
#
#             cv_intrinsics1 = intrinsics1[i].cpu().numpy()
#             cv_intrinsics2 = intrinsics2[i].cpu().numpy()
#
#             cv_i_intrinsics1 = i_intrinsics1[i].cpu().numpy()
#             cv_i_intrinsics2 = i_intrinsics2[i].cpu().numpy()
#
#             E_est_init = estimate_ess_mat_opencv(cv_kp1, cv_nn_kp2, cv_intrinsics1, cv_intrinsics2)
#
#             opt_res = least_squares(loss_fun, E_est_init, jac=loss_fun_jac,
#                                     args=(cv_kp1, cv_nn_kp2, cv_i_intrinsics1, cv_i_intrinsics2), method='lm')
#
#             if opt_res.success:
#                 success_mask[i] = True
#
#                 d_E_o_kp1, d_E_nn_o_kp2 = compute_pose_pt_derivative(opt_res.x, cv_i_intrinsics1, cv_i_intrinsics2,
#                                                                      cv_kp1, cv_nn_kp2)
#
#                 n_i = d_E_o_kp1.shape[0]
#                 d_E_o_kp1 = np.linalg.norm(d_E_o_kp1.reshape(n_i, -1), axis=-1)
#                 d_E_nn_o_kp2 = np.linalg.norm(d_E_nn_o_kp2.reshape(n_i, -1), axis=-1)
#
#                 ds_o_kp1[i, match_mask[i]] = torch.tensor(d_E_o_kp1, dtype=torch.float).to(o_kp1.device)
#                 ds_nn_o_kp2[i, match_mask[i]] = torch.tensor(d_E_nn_o_kp2, dtype=torch.float).to(nn_o_kp2.device)
#
#     return ds_o_kp1, ds_nn_o_kp2, success_mask

# def get_r_t(x):
#     R, jac = cv2.Rodrigues(x[0:3])
#     x2 = np.zeros(3)
#     x2[0:2] = x[3:5]
#     R2, jac2 = cv2.Rodrigues(x2)
#     t = R2[:, 2]
#
#     return R, t

# def compute_epipolar_errors(F, kp1, kp2):
#     errs = []
#     for i in range(0, len(kp1)):
#         x1h = np.ones(3)
#         x1h[0:2] = kp1[i, 0:2]
#         x2h = np.ones(3)
#         x2h[0:2] = kp2[i, 0:2]
#         l = F @ x1h
#         l = l / np.linalg.norm(l[0:2])
#         errs.append(np.abs(x2h.dot(l)))
#     return errs
