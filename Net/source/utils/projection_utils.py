import torch

from Net.source.utils.math_utils import create_coord_grid, sample_grid, to_homogeneous, to_cartesian

"""
Warping functions
"""


def warp_coord_grid_RBT(grid1, depth1, intrinsics1, extrinsics1, shift_scale1, depth2, intrinsics2, extrinsics2,
                        shift_scale2):
    """
    :param grid1: B x H x W x 2
    :param depth1: B x 1 x H x W
    :param intrinsics1: B x 3 x 3
    :param extrinsics1: B x 4 x 4
    :param shift_scale1: B x 4
    :param depth2: B x 1 x H x W
    :param intrinsics2: B x 3 x 3
    :param extrinsics2: B x 4 x 4
    :param shift_scale2: B x 4
    :return: B x H x W x 2, B x 1 x H x W
    """
    b, h, w = grid1.shape[:3]

    grid1_depth = sample_grid(depth1, grid1)

    # Prepare intrinsic matrix by accounting for shift and scale of the image
    c_intrinsics1 = intrinsics1.clone()
    c_intrinsics1[:, :2, 2] -= shift_scale1[:, [1, 0]]

    c_intrinsics1 = c_intrinsics1.inverse()
    c_intrinsics1[:, 0, 0] /= shift_scale1[:, 3]
    c_intrinsics1[:, 1, 1] /= shift_scale1[:, 2]

    # Translate grid cells to their corresponding plane at distance grid_depth from camera
    grid1_3d = to_homogeneous(grid1).view(b, -1, 3).permute(0, 2, 1)  # B x 3 x H * W
    grid1_3d = (c_intrinsics1 @ grid1_3d) * grid1_depth.view(b, 1, -1)
    grid1_3d = to_homogeneous(grid1_3d, dim=1)  # B x 4 x H * W

    # Move 3D points from first camera system to second
    w_grid1_3d = extrinsics2 @ torch.inverse(extrinsics1) @ grid1_3d
    w_grid1_3d = to_cartesian(w_grid1_3d, dim=1)  # B x 3 x H * W

    # Warped depth
    w_grid1_depth = w_grid1_3d[:, 2, :].clone().view(b, 1, h, w)

    # Convert 3D points to their projections on the image plane
    w_grid1_3d = intrinsics2 @ w_grid1_3d  # B x 3 x H * W
    w_grid1 = to_cartesian(w_grid1_3d.permute(0, 2, 1)).view(b, h, w, 2)

    w_grid1 = (w_grid1 - shift_scale2[:, None, None, [1, 0]]) * shift_scale2[:, None, None, [3, 2]]

    # Compose occlusion and depth masks
    w_grid1_depth2 = sample_grid(depth2, w_grid1)
    depth_mask1 = (w_grid1_depth2 > 0) * (torch.abs(w_grid1_depth - w_grid1_depth2) < 0.05)

    return w_grid1, depth_mask1


def warp_image_RBT(image1, depth2, intrinsics2, extrinsics2, shift_scale2, depth1, intrinsics1,
                   extrinsics1, shift_scale1):
    """
    :param image1: B x C x iH x iW
    :param depth2: B x 1 x iH x iW
    :param intrinsics2: B x 3 x 3
    :param extrinsics2: B x 4 x 4
    :param shift_scale2: B x 4
    :param depth1: B x 1 x oH x oW
    :param intrinsics1: B x 3 x 3
    :param extrinsics1: B x 4 x 4
    :param shift_scale1: B x 4
    """
    grid2 = create_coord_grid(depth2.shape).to(image1.device)

    w_grid2, depth_mask2 = warp_coord_grid_RBT(grid2, depth2, intrinsics2, extrinsics2, shift_scale2, depth1,
                                               intrinsics1, extrinsics1, shift_scale1)
    w_image2 = sample_grid(image1, w_grid2) * depth_mask2.float()
    
    w_vis_mask2 = get_visibility_mask(depth1.shape, w_grid2[..., [1, 0]]).unsqueeze(1) * depth_mask2

    return w_image2, w_vis_mask2


def warp_keypoints_RBT(kp1, depth1, intrinsic1, extrinsic1, shift_scale1, depth2, intrinsic2, extrinsic2, shift_scale2):
    """
    :param kp1: B x N x 2, coordinates order is (y, x)
    :param depth1: B x 1 x oH x oW
    :param intrinsic1: B x 3 x 3
    :param extrinsic1: B x 4 x 4
    :param shift_scale1: B x 4
    :param depth2: B x 1 x iH x iW
    :param intrinsic2: B x 3 x 3
    :param extrinsic2: B x 4 x 4
    :param shift_scale2: B x 4
    """
    #  Because warping operates on x,y coordinates we need to swap h and w dimensions
    kp1 = kp1[..., [1, 0]].unsqueeze(1)
    w_kp1, depth_mask1 = warp_coord_grid_RBT(kp1, depth1, intrinsic1, extrinsic1, shift_scale1, depth2, intrinsic2,
                                             extrinsic2, shift_scale2)

    w_kp1 = w_kp1.squeeze(1)[..., [1, 0]]
    depth_mask1 = depth_mask1.squeeze(1).squeeze(1)

    w_vis_kp1_mask = get_visibility_mask(depth2.shape, w_kp1) * depth_mask1

    return w_kp1, w_vis_kp1_mask


def warp_coord_grid_H(grid1, H12):
    """
    :param grid1: B x H x W x 2
    :param H12: B x 3 x 3
    """
    b, h, w, _ = grid1.size()

    # Convert grid to homogeneous coordinates
    grid1 = to_homogeneous(grid1)  # B x H x W x 3

    # Flatten spatial dimensions
    grid1 = grid1.view(b, -1, 3).permute(0, 2, 1)  # B x 3 x H * W

    w_grid1 = torch.matmul(H12, grid1).permute(0, 2, 1)  # B x H * W x 3

    # Convert coordinates from homogeneous to cartesian
    w_grid1 = to_cartesian(w_grid1).view(b, h, w, -1)

    return w_grid1


def warp_image_H(image2_shape, image1, H21):
    """
    :param image2_shape: (b, c, oH, oW)
    :param image1: B x C x iH x iW
    :param H21: B x 3 x 3; A homography to warp coordinates from image2 to image1
    :return w_image: B x C x H x W
    """
    _, _, h, w = image1.shape

    grid2 = create_coord_grid(image2_shape).to(image1.device)
    w_grid2 = warp_coord_grid_H(grid2, H21)

    w_image1 = sample_grid(image1, w_grid2)
    w_vis_mask1 = get_visibility_mask(image1.shape, w_grid2[..., [1, 0]]).unsqueeze(1)

    return w_image1, w_vis_mask1


def warp_keypoints_H(kp1, H12, image2_shape):
    """
    :param kp1: B x N x 2, coordinates order is (y, x)
    :param H12: B x 3 x 3
    :param image2_shape: (b, c, h, w)
    :return B x N x 2
    """
    b, n, _ = kp1.size()

    # Because warping operates on x,y coordinates we need to swap them places
    kp1 = kp1[..., [1, 0]].unsqueeze(1)
    w_kp1 = warp_coord_grid_H(kp1, H12).squeeze(1)[..., [1, 0]]

    w_vis_kp1_mask = get_visibility_mask(image2_shape, w_kp1).float()

    return w_kp1, w_vis_kp1_mask


"""
Support utils
"""


def get_visibility_mask(image_shape, kp):
    """
    :param image_shape: (b, 1, h, w)
    :param kp: B x N x 2 or B x H x W x 2 :type torch.float (y, x) orientation
    :return B x N :type torch.bool
    """
    return kp[..., 0].gt(0) * \
           kp[..., 0].lt(image_shape[2]) * \
           kp[..., 1].gt(0) * \
           kp[..., 1].lt(image_shape[3])

# Legacy code

# import numpy as np
# from scipy.stats import truncnorm

# def sample_homography(shape, perspective, scaling, rotation, translation,
#                       n_scales, n_angles, scaling_amplitude, perspective_amplitude_x,
#                       perspective_amplitude_y, patch_ratio, max_angle):
# # Corners of the output image
# pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
# # Corners of the input patch
# margin = (1 - patch_ratio) / 2
# pts2 = margin + np.array([[0, 0], [0, patch_ratio],
#                           [patch_ratio, patch_ratio], [patch_ratio, 0]])
#
# # Random perspective and affine perturbations
# if perspective:
#     perspective_displacement = truncnorm(-2, 2, 0, perspective_amplitude_y / 2).rvs()
#
#     h_displacement_left = truncnorm(-2, 2, 0., perspective_amplitude_x / 2).rvs()
#     h_displacement_right = truncnorm(-2, 2, 0., perspective_amplitude_x / 2).rvs()
#
#     pts2 += np.array([[h_displacement_left, perspective_displacement],
#                       [h_displacement_left, -perspective_displacement],
#                       [h_displacement_right, perspective_displacement],
#                       [h_displacement_right, -perspective_displacement]])
#
# # Random scaling
# # sample several scales, check collision with borders, randomly pick a valid one
# if scaling:
#     scales = np.concatenate([[1.], truncnorm(-2, 2, 1, scaling_amplitude / 2).rvs(n_scales)], 0)
#     center = np.mean(pts2, axis=0, keepdims=True)
#     scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
#
#     valid = np.arange(n_scales)  # all scales are valid except scale=1
#
#     idx = valid[np.random.randint(0, valid.shape[0])]
#     pts2 = scaled[idx]
#
# # Random translation
# if translation:
#     t_min, t_max = np.amin(pts2, axis=0), np.amin(1 - pts2, axis=0)
#
#     pts2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
#                                      np.random.uniform(-t_min[1], t_max[1])]), axis=0)
#
# # Random rotation
# # sample several rotations, check collision with borders, randomly pick a valid one
# if rotation:
#     angles = np.linspace(-max_angle, max_angle, n_angles)
#     angles = np.concatenate([[0.], angles], axis=0)  # in case no rotation is valid
#     center = np.mean(pts2, axis=0, keepdims=True)
#     rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles),
#                                    np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
#     rotated = np.matmul(np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles + 1, 1, 1]), rot_mat) + center
#     valid = np.arange(n_angles)
#     idx = valid[np.random.randint(0, valid.shape[0])]
#     pts2 = rotated[idx]
#
# # Rescale to actual size
# shape = shape[::-1]  # different convention [y, x]
# pts1 *= np.expand_dims(shape, axis=0)
# pts2 *= np.expand_dims(shape, axis=0)
#
# def ax(p, q):
#     return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]
#
# def ay(p, q):
#     return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]
#
# a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
# p_mat = np.transpose(np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
# homography = np.transpose(np.linalg.lstsq(a_mat, p_mat, rcond=None)[0])
# homography = np.reshape(np.concatenate([homography, np.ones([homography.shape[0], 1])], axis=1), [3, 3])
#
# return homography
# raise NotImplementedError
