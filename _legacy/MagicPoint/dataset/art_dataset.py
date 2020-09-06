from skimage import io

from legacy.MagicPoint.dataset.dataset_pipeline import *

from legacy.common.utils import *

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

primitives_to_draw = [
    'draw_lines',
    'draw_polygon',
    'draw_multiple_polygons',
    'draw_ellipses',
    'draw_star',
    'draw_checkerboard',
    'draw_stripes',
    'draw_cube',
    'gaussian_noise'
]

available_modes = [TRAINING, VALIDATION, TEST]


def collate(batch):
    return {IMAGE: default_collate([d[IMAGE] for d in batch]).float(),
            KEYPOINT_MAP: default_collate([d[KEYPOINT_MAP] for d in batch]).float(),
            MASK: default_collate([d[MASK] for d in batch]).float().unsqueeze(0)}


class ArtificialDataset(Dataset):

    def __init__(self, mode, config):

        assert mode in available_modes

        self.mode = mode
        self.config = config

        primitives = parse_primitives(config['primitives'], primitives_to_draw)

        base_path = Path(config['data_path'], config['name'] + '_{}'.format(config['suffix']))
        base_path.mkdir(parents=True, exist_ok=True)

        # print("Creating base path:", base_path)

        self.images = []
        self.points = []

        for primitive in primitives:
            tar_path = Path(base_path, '{}.tag.gz'.format(primitive))
            # print("Tar file location:", tar_path)

            if not tar_path.exists():
                save_primitive_data(primitive, tar_path, config)

            temp_dir = Path(tempfile.gettempdir(), config['name'] + '_{}'.format(config['suffix']))
            temp_dir.mkdir(parents=True, exist_ok=True)

            # print("Reserving temp dir:", temp_dir)

            tar = tarfile.open(tar_path)
            tar.extractall(path=temp_dir)
            tar.close()

            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)

            e = [str(p) for p in Path(path, 'images', self.mode).iterdir()]

            # print(len(e))

            f = [p.replace('images', 'points') for p in e]
            f = [p.replace('.png', '.npy') for p in f]

            self.images.extend(e[:int(truncate * len(e))])
            self.points.extend(f[:int(truncate * len(f))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        points_path = self.points[index]

        # image in form (h, w)
        image = np.asarray(io.imread(image_path))
        # array in form (n, 2)
        points = np.load(points_path)
        # image in form (h, w)
        mask = np.ones(image.shape)

        # Apply data augmentation
        if self.mode == 'training':
            if self.config['augmentation']['photometric']['enable']:
                image = photometric_augmentation(image, self.config)
            if self.config['augmentation']['homographic']['enable']:
                image, points, mask = homographic_augmentation(image, points, self.config)

        # Convert points to keypoint map
        keypoint_map = get_keypoint_map(image, points)
        image = normalize_image(grayscale2rgb(image))

        item = {IMAGE: image, POINTS: points, KEYPOINT_MAP: keypoint_map, MASK: mask}

        return item



# def estimate_poses(kp1, kp2, kp1_desc, kp2_desc, intrinsic1, intrinsic2, shift_scale1, shift_scale2, pixel_thresholds,
#                    desc_dist_measure):
#     """
#     :param kp1: B x N x 2, :type torch.long
#     :param kp2: B x N x 2, :type torch.long
#     :param kp1_desc: B x N x 2, :type torch.float
#     :param kp2_desc: B x N x 2, :type torch.float
#     :param intrinsic1: B x 3 x 3, :type torch.float
#     :param intrinsic2: B x 3 x 3, :type torch.float
#     :param shift_scale1: B x 4, :type torch.float
#     :param shift_scale2: B x 4, :type torch.float
#     :param pixel_thresholds: P tensor of thresholds, :type torch.float
#     :param desc_dist_measure: descriptor distance measure, :type DescriptorDistance
#     :return: (P x B x 3 x 4, B x N, P x B x N) :type (torch.float, torch.long, torch.bool)
#     """
#     # Convert keypoints from y, x orientation to x, y
#     kp1 = kp1[..., [1, 0]].float()
#     kp2 = kp2[..., [1, 0]].float()
#
#     # Scale and shift image to its original size
#     kp1 = undo_data_transform(kp1, shift_scale1[:, [1, 0]], shift_scale1[:, [3, 2]])
#     kp2 = undo_data_transform(kp2, shift_scale2[:, [1, 0]], shift_scale1[:, [3, 2]])
#
#     # Choose descriptor distance measure
#     if desc_dist_measure is DescriptorDistance.INV_COS_SIM:
#         desc_dist = smooth_inv_cos_sim_mat(kp1_desc, kp2_desc)
#     else:
#         desc_dist = calculate_distance_matrix(kp1_desc, kp2_desc)
#
#     # Perform mutual NN matching
#     nn_desc_value1, nn_desc_ids1 = desc_dist.topk(dim=-1, k=2, largest=False)
#     nn_desc_value2, nn_desc_ids2 = desc_dist.topk(dim=-2, k=2, largest=False)
#
#     # Create mutual matches mask
#     ids1 = torch.arange(0, kp1.shape[1]).repeat(kp1.shape[0], 1).to(nn_desc_ids1.device)
#     nn_ids2 = torch.gather(nn_desc_ids2[:, 0, :], -1, nn_desc_ids1[..., 0])
#     mutual_matches_mask = ids1 == nn_ids2
#
#     # Create Lowe ratio test masks
#     lowe_ratio_mask1 = nn_desc_value1[..., 0] < nn_desc_value1[..., 1] * 0.7
#     lowe_ratio_mask2 = nn_desc_value2[:, 0, :] < nn_desc_value2[:, 1, :] * 0.7
#
#     # Create final match mask
#     nn_lowe_ratio_mask2 = torch.gather(lowe_ratio_mask2, -1, nn_desc_ids1[..., 0])
#     match_mask = mutual_matches_mask * lowe_ratio_mask1 * nn_lowe_ratio_mask2
#
#     # Select keypoints according to established matches
#     nn_kp2 = torch.gather(kp2, 1, nn_desc_ids1[..., 0].unsqueeze(-1).repeat(1, 1, 2))
#
#     est_relative_poses = torch.zeros(pixel_thresholds.shape[0], kp1.shape[0], 3, 4).to(kp1.device)
#     est_inliers = torch.zeros(pixel_thresholds.shape[0], kp1.shape[0], kp1.shape[1], dtype=torch.bool).to(kp1.device)
#
#     for i, p_thresh in enumerate(pixel_thresholds):
#         for j, b_match_mask in enumerate(match_mask):
#             camera1 = intrinsic2camera(intrinsic1[j].cpu())
#             camera2 = intrinsic2camera(intrinsic2[j].cpu())
#
#             if b_match_mask.sum() < 8:
#                 continue
#
#             # Convert pixel coordinates to bearing vectors
#             bearing_vectors1 = camera1.pixel_bearing_many(kp1[j][b_match_mask].cpu().numpy())
#             bearing_vectors2 = camera2.pixel_bearing_many(nn_kp2[j][b_match_mask].cpu().numpy())
#
#             # Convert pixel threshold to angular
#             avg_focal_length = (camera1.focal_x + camera1.focal_y + camera2.focal_x + camera2.focal_y) / 4
#             angular_threshold = torch.atan2(p_thresh.cpu(), avg_focal_length).item()
#
#             T = multiview.relative_pose_ransac(bearing_vectors1, bearing_vectors2, b"STEWENIUS",
#                                                1 - np.cos(angular_threshold), 1000, 0.999)
#             b_inliers = _compute_inliers_bearings(bearing_vectors1, bearing_vectors2, T, angular_threshold)
#
#             T = multiview.relative_pose_optimize_nonlinear(bearing_vectors1[b_inliers], bearing_vectors2[b_inliers],
#                                                            T[:3, 3], T[:3, :3])
#             b_inliers = _compute_inliers_bearings(bearing_vectors1, bearing_vectors2, T, angular_threshold)
#
#             est_relative_poses[i][j] = torch.tensor(T).to(kp1.device)
#             est_relative_poses[i][j][:3, 3] = normalize(est_relative_poses[i][j][:3, 3], dim=-1)
#
#             est_inliers[i][j][b_match_mask] = torch.tensor(b_inliers).to(kp1.device)
#
#     return est_relative_poses, nn_desc_ids1[..., 0], match_mask, est_inliers
# def estimate_poses_gt_opencv(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                              intrinsics1, intrinsics2, shift_scale1, shift_scale2,
#                              pixel_thresholds,
#                              detailed=False):
#     gt_matches_mask, nn_kp_ids1 = \
#         get_best_mutual_gt_matches(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, pixel_thresholds)
#
#     o_kp1, o_kp2 = to_original(kp1, kp2, shift_scale1, shift_scale2)
#     nn_o_kp2 = change_intrinsics(select_kp(o_kp2, nn_kp_ids1), intrinsics2, intrinsics1)
#
#     est_relative_pose = torch.zeros(pixel_thresholds.shape[0], kp1.shape[0], 3, 4).to(kp1.device)
#
#     if detailed:
#         est_inliers_mask = torch.zeros(pixel_thresholds.shape[0], *kp1.shape[:2], dtype=torch.bool).to(kp1.device)
#
#     for i in range(pixel_thresholds.shape[0]):
#         i_gt_matches_mask = gt_matches_mask[i]
#
#         for b in range(kp1.shape[0]):
#             b_gt_matches_mask = i_gt_matches_mask[b]
#
#             if b_gt_matches_mask.sum() < 8:
#                 continue
#
#             b_intrinsics = intrinsics1[b].cpu().numpy()
#
#             cv_kp1 = kp1[b, b_gt_matches_mask].cpu().numpy()
#             nn_cv_kp2 = nn_o_kp2[b, b_gt_matches_mask].cpu().numpy()
#
#             E_est, mask = cv2.findEssentialMat(cv_kp1, nn_cv_kp2, b_intrinsics, method=cv2.RANSAC, prob=0.99999)
#             _, R_est, t_est, inl_mask = cv2.recoverPose(E_est, cv_kp1, nn_cv_kp2, b_intrinsics, mask=mask)
#
#             est_relative_pose[i, b][:3, :3] = torch.tensor(R_est).to(kp1.device).transpose(0, 1)
#             est_relative_pose[i, b][:3, 3] = normalize(torch.tensor(t_est).to(kp1.device), dim=0).view(-1)
#
#             if detailed:
#                 est_inliers_mask[i, b, b_gt_matches_mask] = torch.tensor(inl_mask, dtype=torch.bool).to(
#                     kp1.device).view(-1)
#
#     if detailed:
#         return est_relative_pose, est_inliers_mask, gt_matches_mask, nn_kp_ids1
#     else:
#         return est_relative_pose


# # TODO. Rollback.
# def estimate_poses_gt(kp1, kp2, w_kp1, w_kp2, wv_kp1_mask, wv_kp2_mask, kp1_desc, kp2_desc, intrinsic1, intrinsic2,
#                       shift_scale1, shift_scale2, pixel_thresholds,
#                       desc_dist_measure):
#     """
#     :param kp1: B x N x 2, :type torch.long
#     :param kp2: B x N x 2, :type torch.long
#     :param kp1_desc: B x N x 2, :type torch.float
#     :param kp2_desc: B x N x 2, :type torch.float
#     :param intrinsic1: B x 3 x 3, :type torch.float
#     :param intrinsic2: B x 3 x 3, :type torch.float
#     :param shift_scale1: B x 4, :type torch.float
#     :param shift_scale2: B x 4, :type torch.float
#     :param pixel_thresholds: P tensor of thresholds, :type torch.float
#     :param desc_dist_measure: descriptor distance measure, :type DescriptorDistance
#     :return: (P x B x 3 x 4, B x N, P x B x N) :type (torch.float, torch.long, torch.bool)
#     """
#     # TODO. ROllback.
#     # Calculate pairwise keypoints distances
#     kp_dist1 = calculate_distance_matrix(w_kp1, kp2)
#     kp_dist1 = mask_non_visible_pairs(kp_dist1, wv_kp1_mask, wv_kp2_mask)
#
#     nn_kp_values1, nn_kp_ids1 = kp_dist1.min(dim=-1)
#
#     match_mask = nn_kp_values1.le(3.0)
#     nn_gt_ids = nn_kp_ids1
#     #########
#
#     # Convert keypoints from y, x orientation to x, y
#     kp1 = kp1[..., [1, 0]].float()
#     kp2 = kp2[..., [1, 0]].float()
#
#     # Scale and shift image to its original size
#     kp1 = undo_data_transform(kp1, shift_scale1[:, [1, 0]], shift_scale1[:, [3, 2]])
#     kp2 = undo_data_transform(kp2, shift_scale2[:, [1, 0]], shift_scale1[:, [3, 2]])
#
#     nn_kp2 = torch.gather(kp2, 1, nn_gt_ids.unsqueeze(-1).repeat(1, 1, 2))
#
#     # Choose descriptor distance measure
#     # if desc_dist_measure is DescriptorDistance.INV_COS_SIM:
#     #     desc_dist = calculate_inv_cosine_similarity_matrix(kp1_desc, kp2_desc)
#     # else:
#     #     desc_dist = calculate_distance_matrix(kp1_desc, kp2_desc)
#
#     # Perform mutual NN matching
#     # nn_desc_value1, nn_desc_ids1 = desc_dist.topk(dim=-1, k=2, largest=False)
#     # nn_desc_value2, nn_desc_ids2 = desc_dist.topk(dim=-2, k=2, largest=False)
#
#     # Create mutual matches mask
#     # ids1 = torch.arange(0, kp1.shape[1]).repeat(kp1.shape[0], 1).to(nn_desc_ids1.device)
#     # nn_ids2 = torch.gather(nn_desc_ids2[:, 0, :], -1, nn_desc_ids1[..., 0])
#     # mutual_matches_mask = ids1 == nn_ids2
#
#     # Create Lowe ratio test masks
#     # lowe_ratio_mask1 = nn_desc_value1[..., 0] < nn_desc_value1[..., 1] * 0.7
#     # lowe_ratio_mask2 = nn_desc_value2[:, 0, :] < nn_desc_value2[:, 1, :] * 0.7
#
#     # Create final match mask
#     # nn_lowe_ratio_mask2 = torch.gather(lowe_ratio_mask2, -1, nn_desc_ids1[..., 0])
#     # match_mask = mutual_matches_mask * lowe_ratio_mask1 * nn_lowe_ratio_mask2
#
#     # Select keypoints according to established matches
#     # nn_kp2 = torch.gather(kp2, 1, nn_desc_ids1[..., 0].unsqueeze(-1).repeat(1, 1, 2))
#
#     est_relative_poses = torch.zeros(pixel_thresholds.shape[0], kp1.shape[0], 3, 4).to(kp1.device)
#     est_inliers = torch.zeros(pixel_thresholds.shape[0], kp1.shape[0], kp1.shape[1], dtype=torch.bool).to(kp1.device)
#
#     for i, p_thresh in enumerate(pixel_thresholds):
#         for j, b_match_mask in enumerate(match_mask):
#             camera1 = intrinsic2camera(intrinsic1[j].cpu())
#             camera2 = intrinsic2camera(intrinsic2[j].cpu())
#
#             if b_match_mask.sum() < 8:
#                 continue
#
#             # Convert pixel coordinates to bearing vectors
#             bearing_vectors1 = camera1.pixel_bearing_many(kp1[j][b_match_mask].cpu().numpy())
#             bearing_vectors2 = camera2.pixel_bearing_many(nn_kp2[j][b_match_mask].cpu().numpy())
#
#             # Convert pixel threshold to angular
#             avg_focal_length = (camera1.focal_x + camera1.focal_y + camera2.focal_x + camera2.focal_y) / 4
#             angular_threshold = torch.atan2(p_thresh.cpu(), avg_focal_length).item()
#
#             T = multiview.relative_pose_ransac(bearing_vectors1, bearing_vectors2, b"STEWENIUS",
#                                                1 - np.cos(angular_threshold), 1000, 0.999)
#             b_inliers = _compute_inliers_bearings(bearing_vectors1, bearing_vectors2, T, angular_threshold)
#
#             T = multiview.relative_pose_optimize_nonlinear(bearing_vectors1[b_inliers], bearing_vectors2[b_inliers],
#                                                            T[:3, 3], T[:3, :3])
#             b_inliers = _compute_inliers_bearings(bearing_vectors1, bearing_vectors2, T, angular_threshold)
#
#             est_relative_poses[i][j] = torch.tensor(T).to(kp1.device)
#             est_relative_poses[i][j][:3, 3] = normalize(est_relative_poses[i][j][:3, 3], dim=-1)
#
#             est_inliers[i][j][b_match_mask] = torch.tensor(b_inliers).to(kp1.device)
#
#     # return est_relative_poses, nn_desc_ids1[..., 0], match_mask, est_inliers
#
#     return est_relative_poses, nn_gt_ids, match_mask, est_inliers


# def validate_inliers(kp1, w_kp1, kp2, w_kp2, wv_kp1_mask, wv_kp2_mask, nn_desc_ids, pixel_thresholds, match_mask,
#                      est_inliers):
#     """
#     :param kp1: B x N x 2. :type torch.long
#     :param w_kp1: B x N x 2, :type torch.float
#     :param kp2: B x N x 2, :type torch.long
#     :param w_kp2: B x N x 2, :type torch.float
#     :param wv_kp1_mask: B x N :type torch.bool
#     :param wv_kp2_mask: B x N :type torch.bool
#     :param nn_desc_ids: B x N :type torch.int
#     :param pixel_thresholds: P tensor of thresholds, :type torch.float
#     :param match_mask: B x N :type torch.bool
#     :param est_inliers: P x B x N, :type torch.bool
#     :return: (P x B, P x B, P x B x N, P x B x N, B) :type (torch.float, torch.float, torch.bool, torch.bool, torch.int64)
#     """
#     # Calculate pairwise keypoints distances
#     kp_dist1 = calculate_distance_matrix(w_kp1, kp2)
#     kp_dist1 = mask_non_visible_pairs(kp_dist1, wv_kp1_mask, wv_kp2_mask)
#
#     kp_dist2 = calculate_distance_matrix(kp1, w_kp2)
#     kp_dist2 = mask_non_visible_pairs(kp_dist2, wv_kp1_mask, wv_kp2_mask)
#
#     # Retrieve correspondent keypoints distances
#     nn_kp_values1 = torch.gather(kp_dist1, -1, nn_desc_ids.unsqueeze(-1))
#     nn_kp_values2 = torch.gather(kp_dist2, -1, nn_desc_ids.unsqueeze(-1))
#
#     # Retrieve minimum distance among two re-projections
#     nn_kp_values = torch.cat([nn_kp_values1, nn_kp_values2], dim=-1).min(dim=-1)[0]
#
#     nn_wv_kp2_mask = torch.gather(wv_kp2_mask, -1, nn_desc_ids)
#
#     vis_gt_matches = (match_mask * wv_kp1_mask * nn_wv_kp2_mask).sum(dim=-1)
#
#     est_inliers_ratio = torch.zeros(est_inliers.shape[0], est_inliers.shape[1])
#     vis_gt_inliers_ratio = torch.zeros(est_inliers.shape[0], est_inliers.shape[1])
#
#     correct_inliers = torch.zeros_like(est_inliers)
#     incorrect_inliers = torch.zeros_like(est_inliers)
#
#     for i, (p_thresh, t_inliers) in enumerate(zip(pixel_thresholds, est_inliers)):
#         # Consider only inliers for which we have ground truth in both images
#         v_inliers = t_inliers * wv_kp1_mask * nn_wv_kp2_mask
#
#         correct_inliers[i] = nn_kp_values.le(p_thresh) * v_inliers
#         incorrect_inliers[i] = ~correct_inliers[i] * v_inliers
#
#         est_inliers_ratio[i] = v_inliers.sum(dim=-1).float() / vis_gt_matches.float()
#         vis_gt_inliers_ratio[i] = correct_inliers[i].sum(dim=-1).float() / vis_gt_matches.float()
#
#     return est_inliers_ratio, vis_gt_inliers_ratio, \
#            correct_inliers, incorrect_inliers, vis_gt_matches


# def epipolar_distance_numpy(o_kp1, o_nn_kp2, F):
#     o_kp1_homo = to_homogeneous_coordinates_numpy(o_kp1)
#     o_nn_kp2_homo = to_homogeneous_coordinates_numpy(o_nn_kp2)
#
#     line2 = o_kp1_homo @ F
#     line1 = o_nn_kp2_homo @ F.T
#
#     dist = (line2 * o_nn_kp2_homo).sum(axis=-1)
#     norm = (1.0 / np.maximum(1e-8, np.linalg.norm(line1, axis=-1)) +
#             1.0 / np.maximum(1e-8, np.linalg.norm(line2, axis=-1)))
#
#     norm_dist = np.abs(dist) * norm
#
#     return norm_dist

#
# def vector_to_cross_numpy(vec):
#     T = np.zeros((3, 3))
#
#     T[0, 1] = -vec[2]
#     T[0, 2] = vec[1]
#     T[1, 0] = vec[2]
#     T[1, 2] = -vec[0]
#     T[2, 0] = -vec[1]
#     T[2, 1] = vec[0]
#
#     return T


# def compose_fundamental_matrix(K1, T1, K2, T2):
#     T12 = T2 @ torch.inverse(T1)
#
#     R = T12[:, :3, :3]
#     t = T12[:, :3, 3]
#
#     A = torch.bmm(torch.bmm(K1, R.transpose(1, 2)), t.unsqueeze(-1))
#     C = vector_to_cross(A)
#
#     F = torch.inverse(K2).transpose(1, 2) @ R @ K1.transpose(1, 2) @ C
#     return F

#
# def compose_fundamental_matrix_numpy(K1, T1, K2, T2):
#     T12 = T2 @ np.linalg.inv(T1)
#
#     R = T12[:3, :3]
#     t = T12[:3, 3]
#
#     A = np.dot(np.dot(K1, R.T), t)
#     C = vector_to_cross_numpy(A)
#
#     F = np.linalg.inv(K2).T @ R @ K1.T @ C
#
#     return F

# import cv2
# undo_data_transform, to_homogeneous, to_homogeneous_coordinates_numpy,
# change_intrinsics