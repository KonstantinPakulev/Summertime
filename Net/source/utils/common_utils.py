import numpy as np
import cv2

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


"""
Utils for CV
"""


def torch2cv(img):
    """
    :param img: C x H x W
    """
    img = img.permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)
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
    :param kp1: N x 2
    :param kp2: N x 2
    """
    kp_cv1 = to_cv2_keypoint(kp1)
    kp_cv2 = to_cv2_keypoint(kp2)

    return cv2.drawMatches(img_cv1, kp_cv1, img_cv2, kp_cv2, to_cv2_dmatch(kp_cv1), None)


def to_cv2_dmatch(kp):
    return list(map(lambda x: cv2.DMatch(x, x, x, x), np.arange(0, len(kp))))