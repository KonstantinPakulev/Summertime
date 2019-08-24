import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f"{key}")
            print_dict(value, indent + 1)
        else:
            print("\t" * indent + f"{key:>18} : {value}")


def flat2grid(ids, w):
    """
    :param ids: B x C x N tensor of indices taken from flattened tensor of shape B x C x H x W
    :param w: Last dimension (W) of tensor from which indices were taken
    :return: B x C x N x 2 tensor of coordinates in input tensor B x C x H x W
    """
    o_h = ids // w
    o_w = ids - o_h * w

    o_h = o_h.unsqueeze(-1)
    o_w = o_w.unsqueeze(-1)

    return torch.cat((o_h, o_w), dim=-1)


def plot_figures(figures, nrows=1, ncols=1, size=None):
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
            axes_list.ravel()[ind].imshow(figures[title], cmap=plt.jet())
            axes_list.ravel()[ind].set_title(title)
            axes_list.ravel()[ind].set_axis_off()
        else:
            axes_list.imshow(figures[title], cmap=plt.jet())
            axes_list.set_title(title)
            axes_list.set_axis_off()

    plt.tight_layout()  # optional


"""
Utils for CV
"""


def torch2cv(img, normalize=False, to_rgb=False):
    """
    :param img: C x H x W
    :param normalize: normalize image by max value
    :param to_rgb: convert image to rgb from grayscale
    """
    if normalize:
        img = img / img.max()

    img = img.permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def cv2torch(img):
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    return img


def to_cv2_keypoint(kp):
    """
    :param kp: N x 2
    """
    kp = kp.cpu().detach().numpy()
    kp = list(map(lambda x: cv2.KeyPoint(x[1], x[0], 0), kp))

    return kp


def draw_cv_keypoints(img_cv, kp, color):
    """
    :param img_cv: H x W x C
    :param kp: N x 2
    :param color: tuple (r, g, b)
    """
    kp_cv = to_cv2_keypoint(kp)
    return cv2.drawKeypoints(img_cv, kp_cv, None, color=color)


def draw_cv_matches(img_cv1, img_cv2, kp1, kp2):
    """
    :param img_cv1: H x W x C
    :param img_cv2: H x W x C
    :param kp1: N1 x 2
    :param kp2: N2 x 2
    """
    kp_cv1 = to_cv2_keypoint(kp1)
    kp_cv2 = to_cv2_keypoint(kp2)

    return cv2.drawMatches(img_cv1, kp_cv1, img_cv2, kp_cv2, to_cv2_dmatch(kp1), None)


def to_cv2_dmatch(kp):
    return list(map(lambda x: cv2.DMatch(x, x, x, x), np.arange(0, len(kp))))