import cv2
import numpy as np

import torch
from torchvision.utils import make_grid

from Net.hpatches_dataset import (HOMO,
                                  S_IMAGE1,
                                  S_IMAGE2)
from Net.utils.image_utils import warp_image

# Eval metrics names
LOSS = 'loss'
DET_LOSS = 'det_loss'
DES_LOSS = 'des_loss'
REP_SCORE = 'repeatability_score'
MATCH_SCORE = 'match_score'
NN_MATCH_SCORE = 'nearest_neighbour_match_score'
NNT_MATCH_SCORE = 'nearest_neighbour_thresh_match_score'
NNR_MATCH_SCORE = 'nearest_neighbour_ratio_match_score'
SHOW = 'show'

# Eval metrics inputs
IMAGE_SIZE = 'image_size'
TOP_K = 'top_k'
HOMO_INV = 'homo_inv'
KP_MT = 'kp_match_thresh'
DS_MT = 'des_match_thresh'
DS_MR = 'des_match_ratio'
KP1 = 'kp1'
KP2 = 'kp2'
DESC1 = 'desc1'
DESC2 = 'desc2'

"""
Mappings to process endpoint and calculate metrics
"""


def l_loss(x):
    return x[LOSS]


def l_det_loss(x):
    return x[DET_LOSS]


def l_des_loss(x):
    return x[DES_LOSS]


def l_rep_score(x):
    return repeatability_score(x[KP1], x[KP2], x[HOMO], x[HOMO_INV], x[IMAGE_SIZE], x[TOP_K])


def l_match_score(x):
    return match_score(x[KP1], x[KP2], x[DESC1], x[DESC2], x[TOP_K], x[KP_MT], x[DS_MT], x[DS_MR])


def l_collect_show(x):
    return x[S_IMAGE1], x[S_IMAGE2], x[KP1], x[KP2]


"""
Evaluation functions
"""


def repeatability_score(kp1, kp2, homo, homo_inv, image_size, top_k):
    """
    :param kp1: K x 4
    :param kp2: K x 4
    :param homo: N x 3 x 3
    :param homo_inv: N x 3 x 3
    :param image_size: (N, 1, H, W)
    :param top_k: int
    """
    kp1_map = torch.zeros(image_size).to(kp1.device)
    kp1_map[:, :, kp1[:, 2], kp1[:, 3]] = 1

    kp2_map = torch.zeros(image_size).to(kp2.device)
    kp2_map[:, :, kp2[:, 2], kp2[:, 3]] = 1

    w_kp2_map = warp_image(kp2_map, homo).gt(0).float()
    w_kp1_map = warp_image(kp1_map, homo_inv).gt(0).float()

    kp1_score = (kp1_map * w_kp2_map).sum() / top_k
    kp2_score = (kp2_map * w_kp1_map).sum() / top_k

    return (kp1_score + kp2_score) / 2


def match_score(kp1, kp2, des1, des2, top_k, kp_match_thresh, des_match_thresh, des_match_ratio):
    _, c, s = des1.size()

    des1 = des1.view(c, s)
    des2 = des2.view(c, s)

    # A pair of normalized descriptors will have the maximum value of scalar product if they are the closets
    des_dot = torch.mm(des1.t(), des2)

    v, ids = des_dot.topk(2, dim=-1)

    nn_match_score = nearest_neighbor_match_score(kp1, kp2, ids[:, 0], top_k, kp_match_thresh)
    nn_thresh_match_score = nearest_neighbor_threshold_match_score(kp1, kp2, des1, des2, ids[:, 0], top_k, kp_match_thresh,
                                                             des_match_thresh)
    nn_ratio_match_score = nearest_neighbor_ratio_match_score(kp1, kp2, des1, des2, ids, top_k, kp_match_thresh, des_match_ratio)

    total_match_score = (nn_match_score + nn_thresh_match_score + nn_ratio_match_score) / 3

    return total_match_score, nn_match_score, nn_thresh_match_score, nn_ratio_match_score


def nearest_neighbor_match_score(kp1, kp2, ids, top_k, kp_match_thresh):
    """
    :param kp1: S x 4
    :param kp2: S x 4
    :param ids: S x 1; Ids of closest descriptor of kp2 relative to kp1
    :param top_k: int
    :param kp_match_thresh: int
    """
    kp2 = kp2.index_select(dim=0, index=ids)

    dist = torch.pairwise_distance(kp1[:, 2:].float(), kp2[:, 2:].float())
    correct_matches = dist.le(kp_match_thresh).sum()

    return correct_matches / top_k


def nearest_neighbor_threshold_match_score(kp1, kp2, des1, des2, ids, top_k, kp_match_thresh, des_match_thresh):
    """
    :param kp1: S x 4
    :param kp2: S x 4
    :param ids: S x 1; Ids of closest descriptor of kp2 relative to kp1
    :param des1: C x S
    :param des2: C x S
    :param top_k: int
    :param kp_match_thresh: float
    :param des_match_thresh: float
    """
    des_dist = torch.norm(des1[:, ids] - des2[:, ids], 2)
    thresh_mask = des_dist.le(des_match_thresh)

    kp1 = kp1[thresh_mask, :]
    ids = ids[thresh_mask]

    return nearest_neighbor_match_score(kp1, kp2, ids, top_k, kp_match_thresh)


def nearest_neighbor_ratio_match_score(kp1, kp2, des1, des2, ids, top_k, kp_match_thresh, des_match_ratio):
    """
    :param kp1: S x 4
    :param kp2: S x 4
    :param ids: S x 1; Ids of closest descriptor of kp2 relative to kp1
    :param des1: C x S
    :param des2: C x S
    :param top_k: int
    :param kp_match_thresh: float
    :param des_match_ratio: float
    """
    des_dist1 = torch.norm(des1[:, ids[:, 0]] - des2[:, ids[:, 0]], 2)
    des_dist2 = torch.norm(des1[:, ids[:, 1]] - des2[:, ids[:, 1]], 2)
    dist_ratio = des_dist1 / des_dist2
    thresh_mask = dist_ratio.le(des_match_ratio)

    kp1 = kp1[thresh_mask, :]
    ids = ids[thresh_mask, 0]

    return nearest_neighbor_match_score(kp1, kp2, ids, top_k, kp_match_thresh)


"""
Results visualisation
"""


def torch2cv(img):
    """
    :type img: 1 x C x H x W
    """
    img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)
    return img


def kp2cv(kp):
    """
    :type kp: K x 4
    """
    return cv2.KeyPoint(kp[3], kp[2], 0)


def plot_keypoints(writer, epoch, outputs):
    """
    :param writer: SummaryWriter
    :param epoch: Current train epoch
    :param outputs: list of tuples with all necessary info
    """
    # TODO. Also show matches. You will need descriptors and matching indexes tensor.
    for i, (s_image1, s_image2, kp1, kp2) in enumerate(outputs):
        s_image1 = torch2cv(s_image1)
        s_image2 = torch2cv(s_image2)

        kp1 = kp1.cpu().detach().numpy()
        kp2 = kp2.cpu().detach().numpy()

        kp1 = list(map(kp2cv, kp1))
        kp2 = list(map(kp2cv, kp2))

        s_image1_kp = cv2.drawKeypoints(s_image1, kp1, None, color=(0, 255, 0))
        s_image2_kp = cv2.drawKeypoints(s_image2, kp2, None, color=(0, 255, 0))

        s_image1_kp = s_image1_kp.transpose((2, 0, 1))
        s_image1_kp = torch.from_numpy(s_image1_kp).unsqueeze(0)

        s_image2_kp = s_image2_kp.transpose((2, 0, 1))
        s_image2_kp = torch.from_numpy(s_image2_kp).unsqueeze(0)

        writer.add_image(f"s{i}_keypoints", make_grid(torch.cat((s_image1_kp, s_image2_kp), dim=0)), epoch)
