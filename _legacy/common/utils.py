import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import numpy as np
import os

import torch

# Dataset item dictionary keys
IMAGE = 'image'
WARPED_IMAGE = 'warped_image'

NAME = 'name'
POINTS = 'points'
DEPTH = 'depth'
HOMOGRAPHY = 'homography'

KEYPOINT_MAP = 'keypoint_map'
WARPED_KEYPOINT_MAP = 'warped_keypoint_map'

MASK = 'mask'
WARPED_MASK = 'warped_mask'

# Modes of the dataset
TRAINING = 'training'
VALIDATION = 'validation'
TEST = 'test'


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_checkpoint_path(experiment_config, model_config, epoch):
    base_path = Path(experiment_config['checkpoints_path'])
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path.joinpath(model_config['name'] + '_{}.torch'.format(epoch))


def clear_old_checkpoints(experiment_config):
    base_path = os.path.join(experiment_config['checkpoints_path'])
    if os.path.exists(base_path):
        checkpoints = sorted([os.path.join(base_path, file) for file in os.listdir(base_path)], key=os.path.getmtime)
        for cp in checkpoints[:-experiment_config['keep_checkpoints']]:
            os.remove(cp)


def load_checkpoint(path, map_location=None):
    checkpoint = torch.load(path, map_location)
    return checkpoint['epoch'], checkpoint['model'], checkpoint['optimizer']


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path)


def get_logs_path(experiment_config):
    base_path = Path(experiment_config['logs_path'])
    base_path.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(base_path):
        os.remove(os.path.join(base_path, file))

    return base_path


def plot_images(images, titles=None, cmap='brg', ylabel='', normalize=False, axes=None, dpi=100):
    n = len(images)
    if not isinstance(cmap, list):
        cmap = [cmap] * n
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(6 * n, 6), dpi=dpi)
        if n == 1:
            axes = [axes]
    else:
        if not isinstance(axes, list):
            axes = [axes]
        assert len(axes) == len(images)
    for i in range(n):
        if images[i].shape[-1] == 3:
            images[i] = images[i][..., ::-1]  # BGR to RGB
        axes[i].imshow(images[i], cmap=plt.get_cmap(cmap[i]),
                       vmin=None if normalize else 0,
                       vmax=None if normalize else 1)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():  # remove frame
            spine.set_visible(False)
    axes[0].set_ylabel(ylabel)
    plt.tight_layout()


def grayscale2rgb(image):
    assert len(image.shape) == 2
    return np.stack((image,) * 3, axis=0)


def rgb2grayscale(image):
    assert image.shape[0] == 3
    return image[0]


def normalize_image(image):
    assert image.ravel().max() <= 255 and image.dtype == np.uint8
    image = image.astype(np.float) / 255.0
    return image


def to255scale(image):
    assert image.ravel().max() <= 1
    return (image * 255).astype(np.uint8)


def read_tum_list(base_path, filename):
    tum_list = []

    with open(os.path.join(base_path, filename), 'r') as f:
        lines = f.readlines()[3:]
        for line in lines:
            path = line.rstrip().split()[1]
            tum_list.append(path)

    return tum_list


# relative_translation = (relative_rotation @ -t2.unsqueeze(-1) + t1.unsqueeze(-1)).squeeze(-1)
# relative_translation /= torch.norm(relative_translation, dim=-1).unsqueeze(-1)
#
# print(relative_translation)
# def match_score_with_duplicates(w_kp1, kp2, wv_kp1_mask, wv_kp2_mask, kp_dist_thresholds, kp1_desc, kp2_desc,
#                                 verbose=False):
#     """
#     :param w_kp1: B x N x 2; keypoints on the first image projected to the second
#     :param kp2: B x N x 2; keypoints on the second image
#     :param wv_kp1_mask: B x N; keypoints on the first image which are visible on the second
#     :param wv_kp2_mask: B x N; keypoints on the second image which are visible on the first
#     :param kp_dist_thresholds: float or torch.tensor
#     :param kp1_desc: B x N x C; descriptors for keypoints on the first image
#     :param kp2_desc: B x N x C; descriptors for keypoints on the second image
#     :param verbose: bool
#     """
#     b, n = wv_kp1_mask.shape
#
#     # Calculate pairwise similarity measure
#     desc_sim = calculate_inv_similarity_matrix(kp1_desc, kp2_desc)
#
#     nn_desc_ids = desc_sim.min(dim=-1)[1]
#
#     # Calculate pairwise keypoints distances
#     kp_dist = calculate_distance_matrix(w_kp1, kp2)
#
#     # Remove points that are not visible in both scenes by making their distance larger than maximum
#     max_dist = kp_dist.max() * 2
#     kp_dist = kp_dist + (1 - wv_kp1_mask.float().view(b, n, 1)) * max_dist
#     kp_dist = kp_dist + (1 - wv_kp2_mask.float().view(b, 1, n)) * max_dist
#
#     # Retrieve correspondent keypoints
#     nn_kp_values = torch.gather(kp_dist, -1, nn_desc_ids.view(b, n, 1)).view(b, n)
#
#     num_gt_corr = wv_kp2_mask.sum(dim=-1).float().clamp(min=1e-8)
#
#     match_score_list = torch.empty_like(kp_dist_thresholds)
#
#     if verbose:
#         num_matches_list = torch.empty_like(kp_dist_thresholds)
#         kp_matches_list = torch.zeros([*kp_dist_thresholds.shape, *wv_kp1_mask.shape])
#
#     for i, threshold in enumerate(kp_dist_thresholds):
#         # Threshold correspondences
#         desc_matches = nn_kp_values.le(threshold)
#
#         # Calculate number of matches for each scene
#         num_matches = desc_matches.sum(dim=-1).float()
#
#         m_score = num_matches.float() / num_gt_corr.float()
#         match_score_list[i] = m_score.mean()
#
#         if verbose:
#             num_matches_list[i] = num_matches.mean()
#             kp_matches_list[i] = desc_matches
#
#     if verbose:
#         return match_score_list, num_matches_list, num_gt_corr, nn_desc_ids, kp_matches_list
#     else:
#         return match_score_list

# import numpy as np
# from opensfm.matching import _compute_inliers_bearings

# from Net.source.utils.math_utils import calculate_distance_matrix, calculate_inv_similarity_matrix
# from Net.source.utils.metric_utils import SIM, L2

# def estimate_poses(kp1, kp2, kp1_desc, kp2_desc, intrinsic1, intrinsic2, shift_scale1, shift_scale2, desc_measure=SIM):
#     # Convert keypoints from y, x orientation to x, y
#     kp1 = kp1[..., [1, 0]].float()
#     kp2 = kp2[..., [1, 0]].float()

#     # Scale and shift keypoints to their original positions on the images
#     x_scale1 = shift_scale1[:, 3].view(-1, 1)
#     x_shift1 = shift_scale1[:, 1].view(-1, 1)

#     y_scale1 = shift_scale1[:, 2].view(-1, 1)
#     y_shift1 = shift_scale1[:, 0].view(-1, 1)

#     x_scale2 = shift_scale2[:, 3].view(-1, 1)
#     x_shift2 = shift_scale2[:, 1].view(-1, 1)

#     y_scale2 = shift_scale2[:, 2].view(-1, 1)
#     y_shift2 = shift_scale2[:, 0].view(-1, 1)

#     kp1[..., 0] = kp1[..., 0] / x_scale1 + x_shift1
#     kp1[..., 1] = kp1[..., 1] / y_scale1 + y_shift1

#     kp2[..., 0] = kp2[..., 0] / x_scale2 + x_shift2
#     kp2[..., 1] = kp2[..., 1] / y_scale2 + y_shift2

#     # Calculate pairwise similarity measure
#     if desc_measure == SIM:
#         desc_sim = calculate_inv_similarity_matrix(kp1_desc, kp2_desc)
#     else:
#         desc_sim = calculate_distance_matrix(kp1_desc, kp2_desc)

#     # Perform NN matching
#     nn_desc_value, nn_desc_ids = desc_sim.topk(dim=-1, k=2, largest=False)

#     # Select keypoints according to established matches
#     nn_kp2 = torch.gather(kp2, 1, nn_desc_ids[..., 0].unsqueeze(-1).repeat(1, 1, 2))

#     # Create Lowe ratio test mask
#     threshold = 0.004
#     lowe_ratio_mask = nn_desc_value[..., 0] < nn_desc_value[..., 1] * 0.7

#     relative_poses = torch.zeros(kp1.shape[0], 3, 4).to(kp1.device)
#     inliers = []

#     for i, b_lowe_ratio_mask in enumerate(lowe_ratio_mask):
#         camera1 = intrinsic2camera(intrinsic1[i])
#         camera2 = intrinsic2camera(intrinsic2[i])

#         bearing_vectors1 = camera1.pixel_bearing_many(kp1[i][b_lowe_ratio_mask].numpy())
#         bearing_vectors2 = camera2.pixel_bearing_many(nn_kp2[i][b_lowe_ratio_mask].numpy())

#         T = multiview.relative_pose_ransac(bearing_vectors1, bearing_vectors2,
#                                            b"STEWENIUS", 1 - np.cos(threshold), 1000, 0.999)

#         b_inliers = _compute_inliers_bearings(bearing_vectors1, bearing_vectors2, T, threshold)
#         inliers.append(b_inliers)

#         print(b_inliers.sum())

#         relative_poses[i] = torch.tensor(T)

#     return relative_poses, inliers, nn_desc_ids[...,0], lowe_ratio_mask
# def get_gt_relative_poses(extrinsic1, extrinsic2):
#     R1 = extrinsic1[:, :3, :3]
#     t1 = extrinsic1[:, :3, 3]
#     camera_world_translation1 = -R1.transpose(1, 2) @ t1.unsqueeze(-1)
#
#     R2 = extrinsic2[:, :3, :3]
#     t2 = extrinsic2[:, :3, 3]
#     camera_world_translation2 = -R2.transpose(1, 2) @ t2.unsqueeze(-1)
#
#     relative_world_translation = camera_world_translation2 - camera_world_translation1
#
#     relative_translation = (R1 @ relative_world_translation).squeeze(-1)
#     relative_translation /= torch.norm(relative_translation, dim=-1).unsqueeze(-1)
#
#     relative_rotation = R1 @ R2.transpose(1, 2)
#
#     relative_poses = torch.zeros(extrinsic1.shape[0], 3, 4)
#     relative_poses[:, :3, :3] = relative_rotation
#     relative_poses[:, :3, 3] = relative_translation
#
#     return relative_poses
#
#
# gt_relative_poses = get_gt_relative_poses(batch[dataset_idx][ev.EXTRINSIC1], batch[dataset_idx][ev.EXTRINSIC2])


# def print_pose_estimation(image1_name, image2_name,
#                           est_relative_poses, gt_relative_poses,
#                           est_inliers, prop_match_mask,
#                           est_inliers_ratio, vis_inliers_ratio,
#                           correct_inliers, incorrect_inliers, num_prop_vis_gt_matches,
#                           pixel_thresholds):
#     """
#     :param est_relative_poses: P x B x 3 x 4 :type torch.float
#     :param gt_relative_poses: B x 3 x 4 :type torch.float
#     :param est_inliers: P x B x N, :type torch.bool
#     :param prop_match_mask: P x B :type torch.bool
#     :param est_inliers_ratio: P x B :type torch.float
#     :param vis_inliers_ratio: P x B :type torch.float
#     :param correct_inliers: P x B x N :type torch.bool
#     :param incorrect_inliers: P x B x N :type torch.bool
#     :param num_prop_vis_gt_matches: B :type torch.int64
#     :param pixel_thresholds: P tensor of thresholds, :type torch.float
#     """
#     for b_id in range(prop_match_mask.shape[0]):
#         pair_name = image1_name[b_id] + " and " + image2_name[b_id]
#         print(f"Pair: {pair_name}")
#         print("-" * 66)
#
#         for t_id, thresh in enumerate(pixel_thresholds):
#             print("\t" * 2 + f"Threshold: {thresh} px")
#             print("\t" * 2 + "-" * 18)
#
#             intent = '\t' * 3
#
#             num_est_inliers = est_inliers[t_id][b_id].sum(dim=-1)
#             num_prop_matches = prop_match_mask[b_id].sum(dim=-1)
#
#             num_total_matches = (correct_inliers[t_id][b_id] + incorrect_inliers[t_id][b_id]).sum(dim=-1)
#             num_correct_matches = correct_inliers[t_id][b_id].sum(dim=-1)
#
#             print(intent + f"Estimated inliers/proposed matches: {num_est_inliers}/{num_prop_matches}")
#             print(
#                 intent + f"Visible inliers/ground-truth matches: {num_total_matches}/{num_prop_vis_gt_matches[b_id]} ({est_inliers_ratio[t_id][b_id]:.4})")
#             print(
#                 intent + f"Visible correct inliers/ground-truth matches: {num_correct_matches}/{num_prop_vis_gt_matches[b_id]} ({vis_inliers_ratio[t_id][b_id]:.4})")
#
#             R_err = angle_mat(est_relative_poses[t_id][b_id][:3, :3], gt_relative_poses[b_id][:3, :3])
#             t_err = angle_vec(est_relative_poses[t_id][b_id][:3, 3], gt_relative_poses[b_id][:3, 3])
#
#             print(intent + f"R error: {R_err:.4} degrees")
#             print(intent + f"t error: {t_err:.4} degrees")
#
#             print()
#
#         print()
            # det_criterion = DetectorMSELoss.from_config(self.models_config, self.criterions_config)
            # des_criterion = DescriptorTripletLoss.from_config(self.models_config, self.criterions_config)
            # rel_criterion = ReliabilityLoss(self.models_config[exp.GRID_SIZE])
            # pose_criterion = PoseLoss(self.criterions_config[exp.POSE_LAMBDA])
            #
            # # epipolar_criterion = EpipolarLoss(self.criterions_config[exp.EP_LAMBDA],
            # #                                   self.metric_config[self.mode][exp.PX_THRESH],
            # #                                   DescriptorDistance.INV_COS_SIM)
            # #
            # # re_proj_criterion = ReProjectionLoss(self.criterions_config[exp.EP_LAMBDA],
            # #                                      self.metric_config[self.mode][exp.PX_THRESH],
            # #                                      DescriptorDistance.INV_COS_SIM)
            #
            # # return [det_criterion, des_criterion, rel_criterion, epipolar_criterion]
            # return [det_criterion, des_criterion, rel_criterion, pose_criterion]

# TODO. If you you was fast or ordinary you need to change inheritance
# return [NetVGGDelta.from_config(self.models_config).to(self.device)]
# return [NetVGG.from_config(self.models_config).to(self.device)]

# model = NetDefaultChain(TwoFoldMultiScaleBackboneWrapper(),
#                         MultiScaleDetBranchWrapper(kp_extractor, self.models_config),
#                         DescBranchWrapper(self.models_config), DetLocModuleWrapper())

# MultiScaleDetectorBranch(nms_kernel_size, grid_size)
# DescriptorBranch(model_config[exp.DESCRIPTOR_SIZE])
# kp_extractor, self.models_config

#         DetectionLocalizationBranch(9, 5)
#         # TODO. Replace with parameters after successful testing
# TODO. Notice ^2 was removed
# desc_sim = desc_sim + (1 - w_vis_mask1.float()) * 5
# Prepare visibility mask
# w_vis_mask1 = self.space_to_depth(w_vis_mask1)
# w_vis_mask1 = w_vis_mask1.prod(dim=1).view(b, 1, -1)

# config = criterion_config[exp.DET]
# criterion_config
#         self.criterion = DetectorMSELoss(config[exp.NMS_KERNEL_SIZE], config[exp.TOP_K],
#                                          config[exp.GAUSS_KERNEL_SIZE], config[exp.GAUSS_SIGMA],
#                                          config[exp.LAMBDA])

# import torch
# import torch.nn as nn
# from torch.nn.functional import normalize
#
# import Net.source.experiments.base_experiment as exp
# import Net.source.nn.net.utils.endpoint_utils as f
# import Net.source.nn.net.utils.model_utils as a
#
# from Net.source.nn.net.utils.model_utils import make_vgg_ms_block, make_vgg_ms_detector, make_vgg_ms_descriptor, \
#     multi_scale_nms, multi_scale_softmax, make_vgg_block
#
#
# class NetVGG(nn.Module):
#
#     @staticmethod
#     def from_config(model_config):
#         return NetVGG(model_config[exp.GRID_SIZE],
#                       model_config[exp.DESCRIPTOR_SIZE],
#                       model_config[exp.NMS_KERNEL_SIZE])
#
#     def __init__(self, grid_size, descriptor_size, nms_ks):
#         super().__init__()
#         self.grid_size = grid_size
#         self.descriptor_size = descriptor_size
#         self.nms_ks = nms_ks
#
#         self.conv1, self.score1 = make_vgg_ms_block(1, 64, 1)
#         self.conv2, self.score2 = make_vgg_ms_block(64, 64, 1)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3, self.score3 = make_vgg_ms_block(64, 64, 2)
#         self.conv4, self.score4 = make_vgg_ms_block(64, 64, 2)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv5, self.score5 = make_vgg_ms_block(64, 128, 4)
#         self.conv6, self.score6 = make_vgg_ms_block(128, 128, 4)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv7, self.score7 = make_vgg_ms_block(128, 128, 8)
#         self.conv8, self.score8 = make_vgg_ms_block(128, 128, 8)
#
#         self.detector = make_vgg_ms_detector(128, 256, self.grid_size)
#         self.descriptor = make_vgg_ms_descriptor(128, 256, self.descriptor_size)
#
#     def _forward(self, x):
#         x = self.conv1(x)
#         s1 = self.score1(x)
#
#         x = self.conv2(x)
#         s2 = self.score2(x)
#
#         x = self.pool1(x)
#
#         x = self.conv3(x)
#         s3 = self.score3(x)
#
#         x = self.conv4(x)
#         s4 = self.score4(x)
#
#         x = self.pool2(x)
#
#         x = self.conv5(x)
#         s5 = self.score5(x)
#
#         x = self.conv6(x)
#         s6 = self.score6(x)
#
#         x = self.pool3(x)
#
#         x = self.conv7(x)
#         s7 = self.score7(x)
#
#         x = self.conv8(x)
#         s8 = self.score8(x)
#
#         s9 = self.detector(x)
#
#         multi_scale_scores = torch.cat((s1, s2, s3, s4, s5, s6, s7, s8, s9), dim=1)
#         nms_scores = multi_scale_nms(multi_scale_scores, self.nms_ks)
#         score = multi_scale_softmax(nms_scores)
#
#         desc = self.descriptor(x)
#         desc = normalize(desc)
#
#         endpoint = {
#             f.SCORE: score,
#             f.DESC: desc,
#             # TODO. Remove in general
#             f.MODEL_INFO: {
#                 a.MS_LOG_SCORES: multi_scale_scores,
#                 a.MS_NMS_SCORES: nms_scores,
#             }
#         }
#
#         return x, multi_scale_scores, endpoint
#
#     def forward(self, x):
#         _, _, endpoint = self._forward(x)
#
#         return endpoint
#
#
# class NetVGGFast(nn.Module):
#
#     def __init__(self, grid_size, descriptor_size, nms_ks):
#         super().__init__()
#         self.grid_size = grid_size
#         self.descriptor_size = descriptor_size
#         self.nms_ks = nms_ks
#
#         self.conv1 = make_vgg_block(1, 64)
#         self.conv2, self.score1 = make_vgg_ms_block(64, 64, 1)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = make_vgg_block(64, 64)
#         self.conv4, self.score2 = make_vgg_ms_block(64, 64, 2)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv5 = make_vgg_block(64, 128)
#         self.conv6, self.score3 = make_vgg_ms_block(128, 128, 4)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv7 = make_vgg_block(128, 128)
#         self.conv8, self.score4 = make_vgg_ms_block(128, 128, 8)
#
#         self.descriptor = make_vgg_ms_descriptor(128, 256, self.descriptor_size)
#
#     def _forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#
#         s1 = self.score1(x)
#
#         x = self.pool1(x)
#
#         x = self.conv3(x)
#         x = self.conv4(x)
#
#         s2 = self.score2(x)
#
#         x = self.pool2(x)
#
#         x = self.conv5(x)
#         x = self.conv6(x)
#
#         s3 = self.score3(x)
#
#         x = self.pool3(x)
#
#         x = self.conv7(x)
#         x = self.conv8(x)
#
#         s4 = self.score4(x)
#
#         multi_scale_scores = torch.cat((s1, s2, s3, s4), dim=1)
#         nms_scores = multi_scale_nms(multi_scale_scores, self.nms_ks)
#         score = multi_scale_softmax(nms_scores)
#
#         desc = self.descriptor(x)
#         desc = normalize(desc)
#
#         endpoint = {
#             f.SCORE: score,
#             f.DESC: desc,
#             f.MODEL_INFO: {
#                 a.MS_LOG_SCORES: multi_scale_scores,
#                 a.MS_NMS_SCORES: nms_scores,
#             }
#         }
#
#         return x, multi_scale_scores, endpoint
#
#     def forward(self, x):
#         _, _, endpoint = self._forward(x)
#
#         return endpoint
#
#
# class NetVGGDelta(NetVGG):
# # class NetVGGDelta(NetVGGFast):
#
#
#     @staticmethod
#     def from_config(model_config):
#         return NetVGGDelta(model_config[exp.GRID_SIZE],
#                            model_config[exp.DESCRIPTOR_SIZE],
#                            model_config[exp.NMS_KERNEL_SIZE])
#
#     def __init__(self, grid_size, descriptor_size, nms_ks):
#         super().__init__(grid_size, descriptor_size, nms_ks)
#
#         # self.displacement = make_vgg_block(4, 2, kernel_size=5)
#         self.displacement = make_vgg_block(9, 2, kernel_size=5)
#
#     def _forward(self, x):
#         x, multi_scale_scores, endpoint = super()._forward(x)
#
#         delta = self.displacement(multi_scale_scores)
#
#         endpoint[f.DELTA] = delta.clamp(min=-2.5, max=2.5)
#
#         return x, multi_scale_scores, endpoint
#
#
# class NetVGGDisc(NetVGG):
#
#     @staticmethod
#     def from_config(model_config):
#         return NetVGGDisc(model_config[exp.GRID_SIZE],
#                           model_config[exp.DESCRIPTOR_SIZE],
#                           model_config[exp.NMS_KERNEL_SIZE])
#
#     def __init__(self, grid_size, descriptor_size, nms_ks):
#         super().__init__(grid_size, descriptor_size, nms_ks)
#         self.discriminator = make_vgg_ms_descriptor(128, 256, 2)
#
#     def _forward(self, x):
#         x, endpoint = super()._forward(x)
#
#         disc = self.discriminator(x)
#
#         if self.training:
#             disc = disc[:, 1].unsqueeze(1)
#         else:
#             disc = disc.softmax(dim=1)
#             disc = disc[:, 1].unsqueeze(1)
#
#         endpoint[f.DISC] = disc
#
#         return x, endpoint
# class DetectorWeightedMSELoss(nn.Module):
#
#
#     def __init__(self, nms_kernel_size, top_k, gauss_k_size, gauss_sigma, loss_lambda):
#         super().__init__()
#
#         self.nms_kernel_size = nms_kernel_size
#         self.top_k = top_k
#
#         self.gauss_k_size = gauss_k_size
#         self.gauss_sigma = gauss_sigma
#
#         self.loss_lambda = loss_lambda
#
#
#     def forward(self, score1, w_score2, pos_sim1, w_vis_mask2):
#         b = score1.size(0)
#
#         gt_score1 = prepare_gt_score(w_score2, self.nms_kernel_size, self.top_k)
#         gt_score1 = gaussian_filter(gt_score1, self.gauss_k_size, self.gauss_sigma)
#
#         s_pos_sim1 = F.interpolate(pos_sim1.detach(), scale_factor=8, mode='bilinear', align_corners=False)
#         s_pos_sim1 = 1.0 + s_pos_sim1
#
#         loss = F.mse_loss(score1, gt_score1, reduction='none') * s_pos_sim1 * w_vis_mask2.float() # B x 1 X H x W
#         loss = loss.view(b, -1).sum(dim=-1) / w_vis_mask2.float().view(b, -1).sum(dim=-1).clamp(min=1e-8)  # B
#         loss = self.loss_lambda * loss.mean()
#
#         return loss


# class ReliabilityLoss(nn.Module):
#
#     def __init__(self, grid_size):
#         super().__init__()
#         self.grid_size = grid_size
#         # TODO. Add balance factor. Relative of positive to total approximately 0.35, i.e. 2,85 as balance factor
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')
#
#     def forward(self, disc1, desc1, desc2, w_desc_grid1, w_vis_desc_grid_mask1):
#         b, c, _, _ = desc1.size()
#
#         desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
#         desc2 = desc2.view(b, c, -1).permute((0, 2, 1))
#
#         mdm_mask, nn_desc_ids1 = get_mutual_desc_matches(desc1, desc2, DescriptorDistance.INV_COS_SIM, None)
#         ids = torch.arange(0, mdm_mask.shape[1]).repeat(mdm_mask.shape[0], 1).to(mdm_mask.device)
#         correct_mask = nn_desc_ids1 == ids
#
#         desc_mask1 = mdm_mask * correct_mask
#
#         loss = self.bce(disc1.view(desc1.shape[0], -1), desc_mask1.float())
#         loss = loss.sum(dim=-1) / w_vis_desc_grid_mask1.float().sum(dim=-1).clamp(min=1e-8)  # B
#         loss = loss.mean().clamp(max=0.3)
#
#         return loss
# b, c, hc, wc = desc1.size()
#
# desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
# w_desc1 = sample_descriptors(desc2, w_desc_grid1, self.grid_size)
#
# inv_sim = inv_cos_sim_vec(desc1, w_desc1).view(b, 1, hc, wc)
# reliability = torch.exp(-2 * inv_sim)
#
# loss = F.mse_loss(disc1, reliability, reduction='none') * w_vis_desc_grid_mask1.view(b, 1, hc, wc).float()
# loss = loss.view(b, -1).sum(dim=-1) / w_vis_desc_grid_mask1.float().sum(dim=-1).clamp(min=1e-8)  # B
# loss = loss.mean()
# loss =
# E_init_ests

# R_est, R_gt = E_ests[:, :3], E_gt[:, :3]
# t_est, t_gt = E_ests[:, 3:], E_gt[:, 3:]

# E_ests, E_init_ests, success_mask = self.ess_mat_func(o_kp1, nn_o_kp2, intrinsics1, intrinsics2, gt_matches_mask[0])
# R_est, R_gt = E_ests[:, :3], E_init_ests[:, :3]
# t_est, t_gt = E_ests[:, 3:], E_init_ests[:, 3:]

# R_diff = R_gt - R_est
# t_diff = t_gt - t_est

# loss = (R_diff.norm(dim=-1) + t_diff.norm(dim=-1)) * success_mask.float()

# print("Max loss value: ")
# print(loss.detach().max(dim=-1)[0])

# class PoseLoss_v2(nn.Module):
#
#     def __init__(self, pose_lambda):
#         super().__init__()
#         self.pose_lambda = pose_lambda
#
#     def forward(self, kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                 intrinsics1, intrinsics2, extrinsics1, extrinsics2, shift_scale1, shift_scale2):
#
#         gt_matches_mask, nn_kp_ids1 = get_best_gt_matches(kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, [5.0], mutual=False)
#
#         o_kp1 = to_original(kp1, shift_scale1)
#         o_kp2 = to_original(kp2, shift_scale2)
#         nn_o_kp2 = select_kp(o_kp2, nn_kp_ids1)
#
#         ds_o_kp1, ds_nn_o_kp2, success_mask = estimate_pose_grad(o_kp1, nn_o_kp2, intrinsics1, intrinsics2, gt_matches_mask[0])
#
#         F = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2)
#
#         # TODO. if doesn't work check 1) without pose, 2) check pose, 3) with pose
#
#         importance = (ds_o_kp1 + ds_nn_o_kp2) / 2
#         mask = importance * gt_matches_mask[0].float()
#
#         loss = epipolar_distance(o_kp1, nn_o_kp2, F).clamp(max=5.0) * mask
#         loss = loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
#         loss = self.pose_lambda * (loss * success_mask.float()).mean()
#
#         return loss

# mutual_mask, nn_desc_ids1 = get_mutual_desc_matches(kp1_desc, kp2_desc, self.dd_measure, None)
# threshold_mask = select_gt_matches(nn_desc_ids1, kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, [5.0])
# nn_o_kp2 = select_kp(o_kp2, nn_desc_ids1)

# ep_dist = ( + (1 - gt_matches_mask[0].float()) * 5.0).clamp(max=5.0)

# loss = ep_dist * w_vis_kp1_mask.float()
# loss = loss.sum(dim=-1) / w_vis_kp1_mask.float().sum(dim=-1).clamp(min=1e-8)
# loss = self.ep_lambda * loss.mean()

# class ReProjectionLoss(nn.Module):
#
#     def __init__(self, ep_lambda, px_thresh, dd_measure):
#         super().__init__()
#         self.ep_lambda = ep_lambda
#
#         self.px_thresh = px_thresh
#         self.dd_measure = dd_measure
#
#
#     def forward(self, kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                 kp1_desc, kp2_desc, intrinsics1, intrinsics2, extrinsics1, extrinsics2,
#                 shift_scale1, shift_scale2):
#         mutual_mask, nn_desc_ids1 = get_mutual_desc_matches(kp1_desc, kp2_desc, self.dd_measure, None)
#         threshold_mask = select_gt_matches(nn_desc_ids1, kp1, w_kp1, kp2, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, [5.0])
#
#         correct_mask = mutual_mask * threshold_mask[0]
#
#         o_kp1 = to_original(kp1, shift_scale1)
#         o_w_kp2 = to_original(w_kp2, shift_scale2)
#         nn_o_w_kp2 = select_kp(o_w_kp2, nn_desc_ids1)
#
#         re_proj_dist = (calculate_distance_vec(o_kp1, nn_o_w_kp2) + (1 - correct_mask.float()) * 5.0).clamp(max=5.0)
#
#         loss = re_proj_dist * w_vis_kp1_mask.float()
#         loss = loss.sum(dim=-1) / w_vis_kp1_mask.float().sum(dim=-1).clamp(min=1e-8)
#         loss = self.ep_lambda * loss.mean()
#
#         return loss