# import torch
# import torch.nn as nn
#
# from Net.source.utils.math_utils import normalize_weighted_points, normalize_points, robust_symmetric_epipolar_distance, \
#     to_homogeneous
#
#
# class ModelEstimator(nn.Module):
#
#     def __init__(self):
#         super(ModelEstimator, self).__init__()
#
#     def forward(self, kp1, kp2, weights):
#         """
#         :param kp1: B x N x 3
#         :param kp2: B x N x 3
#         :param weights: B x N
#         """
#         # Implementation of Hartley Eight-Point algorithm
#
#         # Normalization of points with taking their weights into account
#         norm_kp1, norm_transform1 = normalize_weighted_points(kp1, weights)
#         norm_kp2, norm_transform2 = normalize_weighted_points(kp2, weights)
#
#         # Construction of matrix A to find coefficients f_i of fundamental matrix F
#         A = torch.cat((norm_kp1[:, :, 0].unsqueeze(-1) * norm_kp2,
#                        norm_kp1[:, :, 1].unsqueeze(-1) * norm_kp2,
#                        norm_kp2), -1)
#         # Weighting each correspondence
#         A = A * weights.unsqueeze(-1)
#
#         F_estimate = []
#
#         mask = torch.ones(3).to(kp1.device)
#         mask[-1] = 0
#
#         for batch in A:
#             _, _, V = torch.svd(batch)
#             # Solution to the least squares problem which is the singular vector
#             # corresponding to the smallest singular value
#             F = V[:, -1].view(3, 3)
#
#             # Fundamental matrix is rank-deficient so we need to remove the least singular value
#             U, S, V = torch.svd(F)
#             F_proj = U @ torch.diag(S * mask) @ V.t()
#
#             F_estimate.append(F_proj.unsqueeze(0))
#
#         F_estimate = torch.cat(F_estimate, 0)
#         F_estimate = norm_transform1.permute(0, 2, 1).bmm(F_estimate).bmm(norm_transform2)
#
#         return F_estimate
#
#
# class WeightEstimatorNet(nn.Module):
#
#     def __init__(self, input_size):
#         super(WeightEstimatorNet, self).__init__()
#
#         self.estimator = nn.Sequential(
#             nn.Conv1d(input_size, 64, kernel_size=1),
#             nn.InstanceNorm1d(64, affine=True),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(64, 128, kernel_size=1),
#             nn.InstanceNorm1d(128, affine=True),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(128, 1024, kernel_size=1),
#             nn.InstanceNorm1d(1024, affine=True),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(1024, 512, kernel_size=1),
#             nn.InstanceNorm1d(512, affine=True),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(512, 256, kernel_size=1),
#             nn.InstanceNorm1d(256, affine=True),
#             nn.LeakyReLU(),
#
#             nn.Conv1d(256, 1, kernel_size=1),
#
#             nn.Softmax(-1)
#         )
#
#     def forward(self, x):
#         return self.estimator(x).squeeze(1)
#
#
# class NormalizedEightPointNet(nn.Module):
#
#     def __init__(self, num_iter):
#         super(NormalizedEightPointNet, self).__init__()
#         self.num_iter = num_iter
#
#         self.estimator = ModelEstimator()
#
#         self.weights_init = WeightEstimatorNet(4)
#         self.weights_iter = WeightEstimatorNet(6)
#
#     def forward(self, kp1, kp2):
#         """
#         :param kp1: B x N x 2
#         :param kp2: B x N x 2
#         """
#         # Normalize points to range [-1, 1]
#         kp1, norm_transform1 = normalize_points(to_homogeneous(kp1))
#         kp2, norm_transform2 = normalize_points(to_homogeneous(kp2))
#
#         vectors_init = torch.cat(((kp1[:, :, :2] + 1) / 2, (kp2[:, :, :2] + 1) / 2), 2).permute(0, 2, 1)
#         weights = self.weights_init(vectors_init)  # B x N
#
#         F_estimate_init = self.estimator(kp1, kp2, weights)
#         F_estimates = [F_estimate_init]
#
#         for _ in range(1, self.num_iter):
#             residuals = robust_symmetric_epipolar_distance(kp1, kp2, F_estimate_init).unsqueeze(1)
#
#             vectors_iter = torch.cat((vectors_init, weights.unsqueeze(1), residuals), 1)
#             weights = self.weights_iter(vectors_iter)
#
#             F_estimate_iter = self.estimator(kp1, kp2, weights)
#             F_estimates.append(F_estimate_iter)
#
#         return F_estimates, norm_transform1, norm_transform2


# def get_kp_matches(kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, threshold):
#     kp_dist = calculate_distance_matrix(kp1, w_kp2)
#     kp_dist = mask_non_visible_pairs_(kp_dist, w_vis_kp1_mask, w_vis_kp2_mask)
#
#     nn_kp_values, nn_kp_ids = kp_dist.min(dim=-1)
#
#     Create match mask
    # match_mask = nn_kp_values.le(threshold)
    #
    # return nn_kp_ids, match_mask


# def get_kp_matches(kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, threshold=3.0):
#     kp_dist = calculate_distance_matrix(kp1, w_kp2)
#     kp_dist = mask_non_visible_pairs_(kp_dist, w_vis_kp1_mask, w_vis_kp2_mask)
#
#     nn_kp_values, nn_kp_ids = kp_dist.min(dim=-1)
#
#     # Remove duplicate matches in each scene
#     unique_match_mask = calculate_unique_match_mask(nn_kp_values, nn_kp_ids)
#
#     # Create match mask
#     match_mask = nn_kp_values.le(threshold) * unique_match_mask
#
#     return nn_kp_ids, match_mask

# def process_scores(score1, score2, batch, nms_thresh, nms_k_size, top_k, device):
#     """
#     :param score1: B x 1 X H x W :type torch.float
#     :param score2: B x 1 x H x W :type torch.float
#     :param batch: composite batch :type CompositeBatch
#     :param nms_thresh: non-maximum suppression :type float
#     :param nms_k_size: nms kernel size :type int
#     :param top_k: number of keypoints to select :type int
#     :param device :type int
#     """
#     kp1 = localize_keypoints(score1, select_keypoints(score1, nms_thresh, nms_k_size, top_k))
#     kp2 = localize_keypoints(score2, select_keypoints(score2, nms_thresh, nms_k_size, top_k))
#
#     # kp1 = select_keypoints(score1, nms_thresh, nms_k_size, top_k)
#     # kp2 = select_keypoints(score2, nms_thresh, nms_k_size, top_k)
#
#     w_score1_homo, w_score2_homo = None, None
#     w_vis_mask1_homo, w_vis_mask2_homo = None, None
#     w_kp1_homo, w_kp2_homo = None, None
#     wv_kp1_mask_homo, wv_kp2_mask_homo = None, None
#
#     if batch.is_homo:
#         homo12, homo21 = batch.get_homo(d.HOMO12, device), batch.get_homo(d.HOMO21, device)
#
#         score1_homo = batch.split_homo(score1)
#         score2_homo = batch.split_homo(score2)
#
#         w_score1_homo = warp_image_homo(score2_homo.shape, score1_homo, homo21)
#         w_score2_homo = warp_image_homo(score1_homo.shape, score2_homo, homo12)
#
#         w_vis_mask1_homo = warp_image_homo(score2_homo.shape, torch.ones_like(score1_homo).to(score1.device),
#                                            homo21).gt(0)
#         w_vis_mask2_homo = warp_image_homo(score1_homo.shape, torch.ones_like(score2_homo).to(score2.device),
#                                            homo12).gt(0)
#
#         w_kp1_homo = warp_points_homo(batch.split_homo(kp1), homo12)
#         w_kp2_homo = warp_points_homo(batch.split_homo(kp2), homo21)
#
#         wv_kp1_mask_homo = get_visible_keypoints_mask(batch.split_homo(score2).shape, w_kp1_homo)
#         wv_kp2_mask_homo = get_visible_keypoints_mask(batch.split_homo(score1).shape, w_kp2_homo)
#
#     w_score1_r3, w_score2_r3 = None, None
#     w_vis_mask1_r3, w_vis_mask2_r3 = None, None
#     w_kp1_r3, w_kp2_r3 = None, None
#     wv_kp1_mask_r3, wv_kp2_mask_r3 = None, None
#
#     if batch.is_r3:
#         depth1, depth2, \
#         intrinsic1, intrinsic2, \
#         extrinsic1, extrinsic2, \
#         shift_scale1, shift_scale2 = batch.get_r3(d.DEPTH1, device), batch.get_r3(d.DEPTH2, device), \
#                                      batch.get_r3(d.INTRINSIC1, device), batch.get_r3(d.INTRINSIC2, device), \
#                                      batch.get_r3(d.EXTRINSIC1, device), batch.get_r3(d.EXTRINSIC2, device), \
#                                      batch.get_r3(d.SHIFT_SCALE1, device), \
#                                      batch.get_r3(d.SHIFT_SCALE2, device)
#
#         score1_r3 = batch.split_r3(score1)
#         score2_r3 = batch.split_r3(score2)
#
#         w_score1_r3 = warp_image_r3(score2_r3.shape, score1_r3, depth2, intrinsic2, extrinsic2, shift_scale2,
#                                     depth1, intrinsic1, extrinsic1, shift_scale1)
#         w_score2_r3 = warp_image_r3(score1_r3.shape, score2_r3, depth1, intrinsic1, extrinsic1, shift_scale1,
#                                     depth2, intrinsic2, extrinsic2, shift_scale2)
#
#         w_vis_mask1_r3 = warp_image_r3(score2_r3.shape, torch.ones_like(score1_r3).to(score1.device), depth2,
#                                        intrinsic2, extrinsic2, shift_scale2, depth1, intrinsic1, extrinsic1,
#                                        shift_scale1).gt(0)
#         w_vis_mask2_r3 = warp_image_r3(score1_r3.shape, torch.ones_like(score2_r3).to(score2.device), depth1,
#                                        intrinsic1, extrinsic1, shift_scale1, depth2, intrinsic2, extrinsic2,
#                                        shift_scale2).gt(0)
#
#         w_kp1_r3, w_depth_mask1 = warp_points_r3(batch.split_r3(kp1), depth1, intrinsic1, extrinsic1, shift_scale1,
#                                                  depth2, intrinsic2, extrinsic2, shift_scale2)
#         w_kp2_r3, w_depth_mask2 = warp_points_r3(batch.split_r3(kp2), depth2, intrinsic2, extrinsic2, shift_scale2,
#                                                  depth1, intrinsic1, extrinsic1, shift_scale1)
#
#         wv_kp1_mask_r3 = get_visible_keypoints_mask(batch.split_r3(score2).shape, w_kp1_r3) * w_depth_mask1
#         wv_kp2_mask_r3 = get_visible_keypoints_mask(batch.split_r3(score1).shape, w_kp2_r3) * w_depth_mask2
#
#     w_score1 = batch.join(w_score1_homo, w_score1_r3)
#     w_score2 = batch.join(w_score2_homo, w_score2_r3)
#
#     w_vis_mask1 = batch.join(w_vis_mask1_homo, w_vis_mask1_r3)
#     w_vis_mask2 = batch.join(w_vis_mask2_homo, w_vis_mask2_r3)
#
#     w_kp1 = batch.join(w_kp1_homo, w_kp1_r3)
#     w_kp2 = batch.join(w_kp2_homo, w_kp2_r3)
#
#     wv_kp1_mask = batch.join(wv_kp1_mask_homo, wv_kp1_mask_r3)
#     wv_kp2_mask = batch.join(wv_kp2_mask_homo, wv_kp2_mask_r3)
#
#     endpoint = {
#         SCORE1: score1,
#         SCORE2: score2,
#
#         W_SCORE1: w_score1,
#         W_SCORE2: w_score2,
#
#         W_VIS_MASK1: w_vis_mask1,
#         W_VIS_MASK2: w_vis_mask2,
#
#         KP1: kp1,
#         KP2: kp2,
#
#         W_KP1: w_kp1,
#         W_KP2: w_kp2,
#
#         W_VIS_KP1_MASK: wv_kp1_mask,
#         W_VIS_KP2_MASK: wv_kp2_mask
#     }
#
#     return endpoint


# def localize_keypoints_old(score1, kp1):
#     """
#     :param score1: B x 1 X H x W :type torch.float
#     :param kp1: B x N x 2 :type torch.long
#     :return: B x N x 2 :type torch.float
#     """
#     loc_kp1 = torch.zeros_like(kp1, dtype=torch.float)
#
#     for b, (b_score, b_kp) in enumerate(zip(score1, kp1)):
#         for i, i_kp in enumerate(b_kp):
#             y, x = i_kp
#
#             print(y, x)
#
#             y_down, y_up = (y - 1).clamp(min=0), (y + 1).clamp(max=b_score.shape[1] - 1)
#             x_down, x_up = (x - 1).clamp(min=0), (x + 1).clamp(max=b_score.shape[2] - 1)
#
#             dy = (b_score[:, y_up, x] - b_score[:, y_down, x]) / 2
#             dx = (b_score[:, y, x_up] - b_score[:, y, x_down]) / 2
#
#             dyy = b_score[:, y_up, x] - 2 * b_score[:, y, x] + b_score[:, y_down, x]
#             dxy = (b_score[:, y_up, x_up] - b_score[:, y_down, x_up] - b_score[:, y_up, x_down] - b_score[:, y_down, x_down]) / 4
#             dxx = b_score[:, y, x_up] - 2 * b_score[:, y, x] + b_score[:, y, x_down]
#
#             H = torch.tensor([[dyy, dxy],
#                               [dxy, dxx]])
#
#             dD = torch.tensor([[dy],
#                                [dx]])
#
#             print(H)
#             print(dD)
#
#             # Corrections to the keypoint position
#             x_hat = torch.lstsq(-dD, H)[0].view(-1).to(score1.device)
#
#             print(x_hat)
#
#             print(b_score[:, y, x].cpu() + dD.t() @ x_hat.cpu())
#
#             loc_kp1[b, i] = i_kp.float() + x_hat
#
#             print(loc_kp1[b, i])
#
#             break
#         break
#
#     return loc_kp1


# def prepare_gt_score_ver2(score, nms_thresh, k_size, top_k):
#     n, c, h, w = score.size()
#
#     # nms(score, nms_thresh, k_size).view(n, c, -1)
#
#     score = score.view(n, c, -1)
#     _, flat_ids = torch.topk(score, top_k)
#
#     gt_score = torch.zeros_like(score).to(score.device)
#     gt_score = gt_score.scatter(dim=-1, index=flat_ids, value=1.0).view(n, c, h, w)
#
#     return gt_score

# def process_dfe_input(endpoint):
#     # Calculate homography induced matches
#     kp_dist = calculate_distance_matrix(endpoint[KP1], endpoint[W_KP2])
#     kp_dist = mask_non_visible_pairs_(kp_dist, endpoint[WV_KP1_MASK], endpoint[WV_KP2_MASK])
#
#     nn_kp_values, nn_kp_ids = kp_dist.min(dim=-1)
#
#     # Create match mask
#     match_mask = nn_kp_values.le(3.0)
#
#     nn_kp2 = torch.gather(endpoint[KP2], 1, nn_kp_ids.unsqueeze(dim=-1).repeat(1, 1, 2))
#
#     endpoint[MATCH_MASK] = match_mask
#     endpoint[NN_KP2] = nn_kp2
#
#     return endpoint

# """
# DFE math utils
# """
#
#
# def construct_normalize_transform(scale, mean_point):
#     normalize_transform = torch.zeros((mean_point.shape[0], 3, 3)).to(mean_point.device)
#
#     normalize_transform[:, 0, 0] = scale
#     normalize_transform[:, 1, 1] = scale
#     normalize_transform[:, 2, 2] = 1
#
#     normalize_transform[:, 0, 2] = -mean_point[:, 0] * scale
#     normalize_transform[:, 1, 2] = -mean_point[:, 1] * scale
#
#     return normalize_transform
#
#
# def normalize_weighted_points(points, weights):
#     """
#     :param points: B x N x 3
#     :param weights: B x N
#     """
#     weights_sum = weights.sum(-1)
#
#     mean_point = torch.sum(weights.unsqueeze(-1) * points, 1) / weights_sum.unsqueeze(-1)
#     diff = points - mean_point.unsqueeze(1)
#     mean_dist = (weights * diff[:, :, :2].norm(2, -1)).sum(-1) / weights_sum
#
#     scale = 1.4142 / mean_dist
#
#     normalize_transform = construct_normalize_transform(scale, mean_point)
#
#     points = torch.bmm(points, normalize_transform.permute(0, 2, 1))
#
#     return points, normalize_transform
#
#
# def normalize_points(points):
#     """
#     :param points: B x N x 3
#     """
#     mean_point = points.mean(1)
#     diff = points - mean_point.unsqueeze(1)
#     mean_dist = diff[:, :, :2].norm(2, -1).mean(1)
#
#     scale = 1 / mean_dist
#
#     normalize_transform = construct_normalize_transform(scale, mean_point)
#
#     points = torch.bmm(points, normalize_transform.permute(0, 2, 1))
#
#     return points, normalize_transform
#
#
# def symmetric_epipolar_distance(kp1, kp2, F):
#     """
#     :param kp1: B x N x 3; keypoints on the first image in homogeneous coordinates
#     :param kp2: B x N x 3; keypoints on the second image in homogeneous coordinates
#     :param F: B x 3 x 3; Fundamental matrix connecting first and second image planes
#     """
#     epipolar_line1 = torch.bmm(kp1, F)  # B x N x 3
#     epipolar_line2 = torch.bmm(kp2, F.permute(0, 2, 1))
#
#     epipolar_distance = (kp2 * epipolar_line1).sum(dim=2).abs()
#     norm = (1 / epipolar_line1[:, :, :2].norm(2, -1).clamp(min=1e-8) +
#             1 / epipolar_line2[:, :, :2].norm(2, -1).clamp(min=1e-8))
#
#     return epipolar_distance * norm
#
#
# def robust_symmetric_epipolar_distance(kp1, kp2, F, gamma=0.5):
#     """
#     :param kp1: B x N x 3; keypoints on the first image in homogeneous coordinates
#     :param kp2: B x N x 3; keypoints on the second image in homogeneous coordinates
#     :param F: B x 3 x 3; Fundamental matrix connecting first and second image planes
#     :param gamma; float
#     """
#     return torch.clamp(symmetric_epipolar_distance(kp1, kp2, F), max=gamma)
#
#
# def to_homogeneous_coordinates_numpy(t):
#     return np.concatenate((t, np.ones((t.shape[0], 1))), 1)