import numpy as np

from math import pi
from scipy.stats import truncnorm


def flat2mat(H):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return np.reshape(np.concatenate([H, np.ones([H.shape[0], 1])], axis=1), [3, 3])


def sample_homography(shape, perspective=True, scaling=True, rotation=True, translation=True,
                      n_scales=5, n_angles=25, scaling_amplitude=0.2, perspective_amplitude_x=0.2,
                      perspective_amplitude_y=0.2, patch_ratio=0.85, max_angle=pi / 2,
                      allow_artifacts=True, translation_overflow=0.):
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

    return flat2mat(homography)
