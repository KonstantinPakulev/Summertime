import numpy as np
import cv2 as cv

from math import pi
from scipy.stats import truncnorm
from skimage.morphology import erosion

import torch


def sample_homography(shape, perspective=True, scaling=True, rotation=True, translation=True,
                      n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
                      perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi / 2,
                      allow_artifacts=False, translation_overflow=0.):
    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                              [patch_ratio, patch_ratio], [patch_ratio, 0]])

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        perspective_displacement = truncnorm(-2, 2, 0, perspective_amplitude_y / 2).rvs()

        h_displacement_left = truncnorm(-2, 2, 0., perspective_amplitude_x / 2).rvs()
        h_displacement_right = truncnorm(-2, 2, 0., perspective_amplitude_x / 2).rvs()

        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]])

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = np.concatenate([[1.], truncnorm(-2, 2, 1, scaling_amplitude / 2).rvs(n_scales)], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center

        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = np.where(np.logical_and.reduce((scaled >= 0.) & (scaled < 1.), (1, 2)))[0]

        idx = valid[np.random.randint(0, valid.shape[0])]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = np.amin(pts2, axis=0), np.amin(1 - pts2, axis=0)

        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow

        pts2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
                                         np.random.uniform(-t_min[1], t_max[1])]), axis=0)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.], angles], axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles),
                                       np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles + 1, 1, 1]), rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all angles are valid, except angle=0
        else:
            valid = np.where(np.logical_and.reduce((rotated >= 0.) & (rotated < 1.), axis=(1, 2)))[0]
        idx = valid[np.random.randint(0, valid.shape[0])]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= np.expand_dims(shape, axis=0)
    pts2 *= np.expand_dims(shape, axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homography = np.transpose(np.linalg.lstsq(a_mat, p_mat, rcond=None)[0])

    return homography


def mat2flat(H):
    """
    Converts an homography matrix with shape `[1, 3, 3]` to its corresponding flattened
    homography transformation with shape `[1, 8]`.
    """
    H = np.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def mat2flat_torch(H):
    H = H.reshape((-1, 1, 9))
    return (H / H[:, :, 8:9])[:, :, :8]


def invert_homography(H):
    """
    Computes the inverse transformation for a flattened homography transformation.
    """
    return mat2flat(np.linalg.inv(flat2mat(H)))


def invert_homography_torch(H):
    return mat2flat_torch(torch.inverse(flat2mat_torch(H)))


def flat2mat(H):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return np.reshape(np.concatenate([H, np.ones([H.shape[0], 1])], axis=1), [-1, 3, 3])


def flat2mat_torch(H):
    return torch.cat([H, torch.ones([H.shape[0], H.shape[1], 1])], dim=-1).reshape((-1, 3, 3))


def warp_points(points, homography):
    H = np.expand_dims(homography, axis=0) if len(homography.shape) == 1 else homography

    # Get the points to the homogeneous format
    num_points = np.shape(points)[0]
    points = points[:, ::-1]
    points = np.concatenate([points, np.ones([num_points, 1], dtype=np.float32)], -1)

    # Apply the homography
    H_inv = np.transpose(flat2mat(invert_homography(H)))

    warped_points = np.tensordot(points, H_inv, [1, 0])
    warped_points = warped_points[:, :2, :] / warped_points[:, 2:, :]
    warped_points = np.transpose(warped_points, [2, 0, 1])[:, :, ::-1]

    return warped_points[0] if len(homography.shape) == 1 else warped_points


def warp_points_torch(points, homography):
    num_points = points.shape[0]
    points = points.flip(dims=[1])
    points = torch.cat([points, torch.ones([num_points, 1])], dim=-1)

    H_inv = flat2mat_torch(invert_homography_torch(homography)).permute((2, 1, 0))

    warped_points = torch.tensordot(points, H_inv, dims=([1], [0]))
    warped_points = warped_points[:, :2, :] / warped_points[:, 2:, :]
    warped_points = warped_points.permute(2, 0, 1).flip(dims=(1, 2))

    return warped_points


def filter_points(points, shape):
    mask = (points >= 0) & (points <= np.array(shape) - 1)
    return points[np.logical_and.reduce(mask, -1)]


def compute_valid_mask(image_shape, homography, erosion_radius=0):
    warped_mask = cv.warpPerspective(np.ones(image_shape, dtype=np.uint8), flat2mat(homography)[0],
                                     image_shape[::-1], flags=cv.WARP_INVERSE_MAP)

    if erosion_radius > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
        warped_mask = erosion(warped_mask, kernel)

    return warped_mask
