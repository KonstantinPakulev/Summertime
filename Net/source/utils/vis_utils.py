import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator

import torch

"""
OpenCV plotting/support functions
"""


def draw_cv_keypoints(image, kp, batch_id=0, vis_kp_mask=None, color=(0, 255, 0)):
    """
    :param image: B x C x H x W, :type torch.tensor
    :param kp: B x N x 2
    :param batch_id: int
    :param vis_kp_mask: B x N
    :param color: tuple (r, g, b)
    """
    cv_image = torch2cv(image[batch_id])

    if vis_kp_mask is not None:
        kp = kp[batch_id][vis_kp_mask[batch_id]]
    else:
        kp = kp[batch_id]

    cv_kp = to_cv2_keypoint(kp)

    return cv2.drawKeypoints(cv_image, cv_kp, None, color=color)


def draw_cv_matches(image1, image2, kp1, kp2, matches, match_mask, batch_id=0, match_color=(0, 255, 0),
                    single_point_color=(255, 0, 0)):
    """
    :param image1: B x C x H x W, :type torch.tensor
    :param image2: B x C x H x W, :type torch.tensor
    :param kp1: B x N x 2, :type torch.int64
    :param kp2: B x N x 2, :type torch.int64
    :param matches: B x N, :type torch.bool
    :param match_mask: B x N, :type torch.bool
    :param batch_id: int
    :param match_color: (r, g, b) :type tuple
    :param single_point_color: (r, g, b) :type tuple
    """
    cv_image1 = torch2cv(image1[batch_id])
    cv_image2 = torch2cv(image2[batch_id])

    cv_kp1 = to_cv2_keypoint(kp1[batch_id])
    cv_kp2 = to_cv2_keypoint(kp2[batch_id])

    matches = to_cv2_dmatch(kp1.shape[1], matches[batch_id])
    match_mask = match_mask[batch_id].detach().cpu().numpy().tolist()

    return cv2.drawMatches(cv_image1, cv_kp1, cv_image2, cv_kp2,
                           matches, None,
                           matchColor=match_color,
                           singlePointColor=single_point_color,
                           matchesMask=match_mask)


"""
Matplotlib plotting functions
"""


def plot_reproj_error_hist(nn_kp_values, matches_mask, batch_id=0):
    """
    :param nn_kp_values: B x N
    :param matches_mask: B x N
    :param batch_id: int
    """
    reproj_errors = nn_kp_values[batch_id][matches_mask[batch_id]].cpu().numpy()

    plt.hist(reproj_errors)


def plot_pose_precision_thresh_curve(pose_precision, pose_type="Pose", t_range=180):
    angles = np.linspace(1, t_range, num=t_range)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4, color='r')

    ax.set_title(f"{pose_type} precision/threshold curve", fontsize=19.0)
    ax.set_xlabel("Threshold [degrees]", fontsize=15.0)
    ax.set_ylabel(f"{pose_type} precision", fontsize=17.0)

    for method_name, precision in pose_precision.items():
        if 'W' in method_name:
            ax.plot(angles, precision[:t_range], '--', linewidth=2, label=method_name)
        elif 'B' in method_name:
            ax.plot(angles, precision[:t_range], '-.', linewidth=2, label=method_name)
        else:
            ax.plot(angles, precision[:t_range], linewidth=2, label=method_name)

    ax.legend()
    ax.grid()


def plot_mean_matching_accuracy(eval_results):
    fig, axes = plt.subplots(1, len(eval_results), figsize=(7 * len(eval_results), 6))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].set_ylabel("MMA", fontsize=21.0)

    for i, (ax, (dataset_name, eval_summary)) in enumerate(zip(axes, eval_results)):
        xticks = np.arange(1, 11)

        for key, value in eval_summary.items():
            ax.plot(xticks, value, linewidth=3, label=key)

        ax.set_title(dataset_name, fontsize=23.0)
        ax.set_xlabel('Threshold [px]', fontsize=21.0)

        ax.set_xlim([1, 10])
        ax.set_ylim([0, 1])
        ax.set_xticks(xticks)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax.grid()
        ax.legend()


def plot_precision_recall_curve(precision, recall):
    """
    :param precision: N
    :param recall: N
    """
    _, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(precision, recall)

    ax.set_xlabel('precision', fontsize=25.0)
    ax.set_ylabel('recall', fontsize=25.0)

    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=20)


def plot_figures(figures, nrows=1, ncols=1, size=(18, 18)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axes_list = plt.subplots(ncols=ncols, nrows=nrows, figsize=size)
    for ind, title in zip(range(len(figures)), figures):
        if nrows * ncols != 1:
            axes_list.ravel()[ind].imshow(figures[title], cmap='gray')
            axes_list.ravel()[ind].set_title(title)
            axes_list.ravel()[ind].set_axis_off()
        else:
            axes_list.imshow(figures[title], cmap='gray')
            axes_list.set_title(title)
            axes_list.set_axis_off()

    plt.tight_layout()  # optional


"""
Support functions
"""


def torch2cv(img, normalize=False, to_rgb=False):
    """
    :param img: C x H x W
    :param normalize: normalize image by max value
    :param to_rgb: convert image to rgb from grayscale
    """
    if normalize:
        img = img / img.max()

    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 1:
        img = img[:, :, 0]

    return img


def cv2torch(img):
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    return img


def to_cv2_keypoint(kp):
    """
    :param kp: N x 2
    """
    if torch.is_tensor(kp):
        kp = kp.detach().cpu().numpy()

    kp = list(map(lambda x: cv2.KeyPoint(x[1], x[0], 0), kp))

    return kp


def to_cv2_dmatch(num_kp, matches):
    """
    :param num_kp: int
    :param matches: N
    """
    matches = matches.detach().cpu().numpy()
    return list(map(lambda x: cv2.DMatch(x[0], x[1], 0, 0), zip(np.arange(0, num_kp), matches)))


# Legacy code

# def draw_cv_grid(cv_image, grid_size, grid_color=(25, 25, 25)):
#     """
#     :param cv_image: H x W x C, :type numpy.uint8
#     :param grid_size: int
#     :param grid_color: (r, g, b) :type tuple
#     """
#     h, w = cv_image.shape[:2]
#
#     x, y = np.arange(0, w, step=grid_size), np.arange(0, h, step=grid_size)
#
#     for i in x:
#         cv_image = cv2.line(cv_image, (i, 0), (i, h - 1), grid_color, thickness=1)
#
#     for i in y:
#         cv_image = cv2.line(cv_image, (0, i), (w - 1, i), grid_color, thickness=1)
#
#     return cv_image


# def draw_neigh_mask(image2, w_desc_grid1, neigh_mask_ids, desc_shape, grid_size, w_desc_id):
#     w_desc_point = w_desc_grid1[None, w_desc_id].cpu().numpy()
#     w_desc_neigh_id = neigh_mask_ids[w_desc_id]
#
#     wc = desc_shape[-1]
#     w_desc_neigh_points = flat2grid(w_desc_neigh_id, wc).cpu().numpy() * grid_size
#
#     cv_image = draw_cv_keypoints(image2, w_desc_point, (255, 0, 0))
#     cv_image = draw_cv_keypoints(image2, w_desc_neigh_points, (0, 255, 0))
#
#     return cv_image
