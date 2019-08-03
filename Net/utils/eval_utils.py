import cv2
import numpy as np

import torch
from torchvision.utils import make_grid

from Net.hpatches_dataset import (S_IMAGE1,
                                  S_IMAGE2)
from Net.utils.image_utils import warp_keypoints

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
KP_MT = 'kp_match_thresh'
DS_MT = 'des_match_thresh'
DS_MR = 'des_match_ratio'
KP1 = 'kp1'
KP2 = 'kp2'
W_KP1 = 'w_kp1'
W_KP2 = 'w_kp2'
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
    return repeatability_score(x[KP1], x[W_KP2], x[KP2], x[TOP_K], x[KP_MT])[0]


def l_match_score(x):
    return match_score(x[KP1], x[W_KP2], x[KP2], x[DESC1], x[DESC2], x[TOP_K], x[KP_MT], x[DS_MT], x[DS_MR])


def l_collect_show(x):
    return x[S_IMAGE1], x[S_IMAGE2], x[KP1], x[W_KP2], x[KP2], x[DESC1], x[DESC2], x[TOP_K], x[KP_MT]


"""
Evaluation functions
"""


def determine_matched_keypoints(kp1, w_kp2):
    """
    :param kp1: K x 4
    :param w_kp2: K x 4
    """
    s_kp1 = kp1[:, 2:].float()  # K x 2
    s_kp2 = w_kp2[:, 2:].float()  # K x 2

    s_kp1 = s_kp1.view(s_kp1.size(0), 1, s_kp1.size(1))
    s_kp2 = s_kp2.view(1, s_kp2.size(0), s_kp2.size(1))

    dist_kp = torch.norm(s_kp1 - s_kp2, p=2, dim=-1)

    return dist_kp.min(dim=-1)


def repeatability_score(kp1, w_kp2, kp2, top_k, kp_match_thresh):
    """
    :param kp1: K x 4
    :param w_kp2: K x 4
    :param kp2: K x 4
    :param homo21: N x 3 x 3
    :param top_k: int
    :param kp_match_thresh: float
    """
    _, ids1 = determine_matched_keypoints(kp1, w_kp2)

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, ids1, top_k, kp_match_thresh)


def match_score(kp1, w_kp2, kp2, desc1, desc2, top_k, kp_match_thresh, des_match_thresh, des_match_ratio):
    _, ids = determine_nearest_neighbours(desc1, desc2)

    nn_match_score, _, _ = nearest_neighbor_match_score(kp1, w_kp2, kp2, ids[:, 0], top_k, kp_match_thresh)
    nn_thresh_match_score, _, _ = nearest_neighbor_threshold_match_score(kp1, w_kp2, kp2, desc1, desc2, ids[:, 0],
                                                                         top_k, kp_match_thresh, des_match_thresh)
    nn_ratio_match_score, _, _ = nearest_neighbor_ratio_match_score(kp1, w_kp2, kp2, desc1, desc2, ids,
                                                                    top_k, kp_match_thresh, des_match_ratio)

    total_match_score = (nn_match_score + nn_thresh_match_score + nn_ratio_match_score) / 3

    return [total_match_score, nn_match_score, nn_thresh_match_score, nn_ratio_match_score]


def determine_nearest_neighbours(desc1, desc2):
    """
    :param desc1: S x C
    :param desc2: S x C
    :return:
    """
    # A pair of normalized descriptors will have the maximum value of scalar product if they are the closets
    des_dot = torch.mm(desc1, desc2.t())

    return des_dot.topk(2, dim=-1)


def nearest_neighbor_match_score(kp1, w_kp2, kp2, ids, top_k, kp_match_thresh):
    """
    :param kp1: S x 4; Keypoints from score1
    :param w_kp2: S x 4; Keypoints from score2 warped to score1
    :param kp2: Keypoints from score2
    :param ids: S x 1; Ids of closest descriptor of kp2 relative to kp1
    :param top_k: int
    :param kp_match_thresh: int
    """
    w_kp2 = w_kp2.index_select(dim=0, index=ids)
    kp2 = kp2.index_select(dim=0, index=ids)

    dist = torch.pairwise_distance(kp1[:, 2:].float(), w_kp2[:, 2:].float())
    correct_matches = dist.le(kp_match_thresh)

    score = correct_matches.sum().float() / top_k

    kp1 = kp1[correct_matches, :]
    kp2 = kp2[correct_matches, :]

    return score, kp1, kp2


def nearest_neighbor_threshold_match_score(kp1, w_kp2, kp2, desc1, desc2, ids, top_k, kp_match_thresh,
                                           des_match_thresh):
    """
    :param kp1: S x 4; Keypoints from score1
    :param w_kp2: S x 4; Keypoints from score2 warped to score1
    :param kp2: Keypoints from score2
    :param ids: S x 1; Ids of closest descriptor of kp2 relative to kp1
    :param desc1: C x S
    :param desc2: C x S
    :param top_k: int
    :param kp_match_thresh: float
    :param des_match_thresh: float
    """
    desc2 = desc2.index_select(dim=0, index=ids)

    des_dist = torch.pairwise_distance(desc1, desc2)
    thresh_mask = des_dist.le(des_match_thresh)

    kp1 = kp1[thresh_mask, :]
    ids = ids[thresh_mask]

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, ids, top_k, kp_match_thresh)


def nearest_neighbor_ratio_match_score(kp1, w_kp2, kp2, desc1, desc2, ids, top_k, kp_match_thresh, des_match_ratio):
    """
    :param kp1: S x 4; Keypoints from score1
    :param w_kp2: S x 4; Keypoints from score2 warped to score1
    :param kp2: Keypoints from score2
    :param ids: S x 1; Ids of closest descriptor of kp2 relative to kp1
    :param desc1: C x S
    :param desc2: C x S
    :param top_k: int
    :param kp_match_thresh: float
    :param des_match_ratio: float
    """
    desc2_b = desc2.index_select(dim=0, index=ids[:, 0])
    desc2_c = desc2.index_select(dim=0, index=ids[:, 1])

    des_dist1 = torch.pairwise_distance(desc1, desc2_b)
    des_dist2 = torch.pairwise_distance(desc1, desc2_c)
    dist_ratio = des_dist1 / des_dist2
    thresh_mask = dist_ratio.le(des_match_ratio)

    kp1 = kp1[thresh_mask, :]
    ids = ids[thresh_mask, 0]

    return nearest_neighbor_match_score(kp1, w_kp2, kp2, ids, top_k, kp_match_thresh)


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


def cv2torch(img):
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    return img


def to_cv2_keypoint(kp):
    """
    :type kp: K x 4
    """
    kp = kp.cpu().detach().numpy()
    kp = list(map(lambda x: cv2.KeyPoint(x[3], x[2], 0), kp))

    return kp


def to_cv2_dmatch(kp):
    return list(map(lambda x: cv2.DMatch(x, x, x, x), np.arange(0, len(kp))))


# TODO. Metrics and visualisation results should be for both images. Find intersection in case of visualization
def plot_keypoints(writer, epoch, outputs):
    """
    :param writer: SummaryWriter
    :param epoch: Current train epoch
    :param outputs: list of tuples with all necessary info
    """
    for i, (s_image1, s_image2, kp1, w_kp2, kp2, desc1, desc2, top_k, kp_match_thresh) in enumerate(outputs):
        s_image1 = torch2cv(s_image1)
        s_image2 = torch2cv(s_image2)

        """
        Detected keypoints
        """
        d_kp1 = to_cv2_keypoint(kp1)
        d_kp2 = to_cv2_keypoint(kp2)

        s_image1_kp = cv2.drawKeypoints(s_image1, d_kp1, None, color=(0, 255, 0))
        s_image2_kp = cv2.drawKeypoints(s_image2, d_kp2, None, color=(0, 255, 0))

        s_image1_kp = cv2torch(s_image1_kp).unsqueeze(0)
        s_image2_kp = cv2torch(s_image2_kp).unsqueeze(0)

        """
        Coordinates matched keypoints
        """
        _, km_kp1, km_kp2 = repeatability_score(kp1, w_kp2, kp2, top_k, kp_match_thresh)

        km_kp1 = to_cv2_keypoint(km_kp1)
        km_kp2 = to_cv2_keypoint(km_kp2)

        kp_matches = cv2.drawMatches(s_image1, km_kp1, s_image2, km_kp2, to_cv2_dmatch(km_kp1), None)
        kp_matches = cv2torch(kp_matches)

        """
        Descriptors matched keypoints 
        """
        _, ids = determine_nearest_neighbours(desc1, desc2)
        _, dm_kp1, dm_kp2 = nearest_neighbor_match_score(kp1, w_kp2, kp2, ids[:, 0], top_k, kp_match_thresh)

        dm_kp1 = to_cv2_keypoint(dm_kp1)
        dm_kp2 = to_cv2_keypoint(dm_kp2)

        desc_matches = cv2.drawMatches(s_image1, dm_kp1, s_image2, dm_kp2, to_cv2_dmatch(dm_kp1), None)
        desc_matches = cv2torch(desc_matches)

        writer.add_image(f"s{i}_keypoints", make_grid(torch.cat((s_image1_kp, s_image2_kp), dim=0)),
                         epoch)
        writer.add_image(f"s{i}_keypoints_matches", kp_matches, epoch)
        writer.add_image(f"s{i}_descriptor_matches", desc_matches, epoch)
