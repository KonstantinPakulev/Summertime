# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:20
# @Author  : xylon
import os
import sys
import cv2
import torch
import random
import argparse
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from legacy.ST_Net.utils.common_utils import gct
from legacy.ST_Net.model.st_net_vgg import STNetVGGModule
from legacy.ST_Net.model.st_des_vgg import STDesVGGModule
from legacy.ST_Net.model.st_det_vgg import STDetVGGModule
from legacy.ST_Net.config import cfg


def distance_matrix_vector(anchor, positive):
    """
    Given batch of anchor descriptors and positive descriptors calculate distance matrix
    :param anchor: (B, 128)
    :param positive: (B, 128)
    :return:
    """
    eps = 1e-8
    FeatSimi_Mat = 2 - 2 * torch.mm(anchor, positive.t())  # [0, 4]
    FeatSimi_Mat = FeatSimi_Mat.clamp(min=eps, max=4.0)
    FeatSimi_Mat = torch.sqrt(FeatSimi_Mat)  # euc [0, 2]
    return FeatSimi_Mat

def nearest_neighbor_distance_ratio_match(des1, des2, kp2, threshold):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    sorted, indices = des_dist_matrix.sort(dim=-1)
    Da, Db, Ia = sorted[:, 0], sorted[:, 1], indices[:, 0]
    DistRatio = Da / Db
    predict_label = DistRatio.lt(threshold)
    nn_kp2 = kp2.index_select(dim=0, index=Ia.view(-1))
    return predict_label, nn_kp2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--save", default=None, type=str)  # save path
    parser.add_argument("--imgpath", default=None, type=str)  # image path
    parser.add_argument("--resume", default=None, type=str)  # model path
    args = parser.parse_args()

    print(f"{gct()} : start time")

    random.seed(cfg.PROJ.SEED)
    torch.manual_seed(cfg.PROJ.SEED)
    np.random.seed(cfg.PROJ.SEED)

    print(f"{gct()} : model init")
    det = STDetVGGModule(
        cfg.MODEL.GRID_SIZE,
        cfg.TRAIN.NMS_THRESH,
        cfg.TRAIN.NMS_KSIZE,
        cfg.TRAIN.TOPK,
        cfg.MODEL.GAUSSIAN_KSIZE,
        cfg.MODEL.GAUSSIAN_SIGMA,
    )
    des = STDesVGGModule(8, 128)
    model = STNetVGGModule(
        det, des
    )

    print(f"{gct()} : to device")
    device = torch.device("cpu")
    model = model.to(device)
    resume = args.resume
    print(f"{gct()} : in {resume}")
    checkpoint = torch.load(resume, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])


    def to_cv2_kp(kp):
        # kp is like [batch_idx, y, x, channel]
        return cv2.KeyPoint(kp[3], kp[2], 0)


    def to_cv2_dmatch(m):
        return cv2.DMatch(m, m, m, m)


    def reverse_img(img):
        """
        reverse image from tensor to cv2 format
        :param img: tensor
        :return: RBG image
        """
        img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)  # change to opencv format
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # gray to rgb
        return img


    ###############################################################################
    # detect and compute
    ###############################################################################
    img_path = args.imgpath
    image_paths = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.ppm')]
    detections = []

    save = args.save
    detections_path = os.path.join(save, "material/detections")
    matches_path = os.path.join(save, "material/matches")

    if not os.path.exists(detections_path):
        os.mkdir(detections_path)

    if not os.path.exists(matches_path):
        os.mkdir(matches_path)

    for i, p in enumerate(image_paths):
        kp, desc, img = model.detectAndCompute(p, device, (450, 600))
        detections.append((kp, desc, img))

        keypoints = list(map(to_cv2_kp, kp))
        image_detections = cv2.drawKeypoints(reverse_img(img), keypoints, None, color=(0, 255, 0))

        cv2.imwrite(os.path.join(detections_path, str(i) + ".png"), image_detections)

    for i, (a, b) in enumerate(zip(detections[:-1], detections[1:])):
        kp1, des1, img1 = a
        kp2, des2, img2 = b

        predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(des1, des2, kp2, 0.7)
        idx = predict_label.nonzero().view(-1)

        mkp1 = kp1.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
        mkp2 = nn_kp2.index_select(dim=0, index=idx.long())  # predict match keypoints in I2

        img1, img2 = reverse_img(img1), reverse_img(img2)

        keypoints1 = list(map(to_cv2_kp, mkp1))
        keypoints2 = list(map(to_cv2_kp, mkp2))

        DMatch = list(map(to_cv2_dmatch, np.arange(0, len(keypoints1))))

        # matches1to2	Matches from the first image to the second one, which means that
        # keypoints1[i] has a corresponding point in keypoints2[matches[i]] .

        image_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, DMatch, None)
        cv2.imwrite(os.path.join(matches_path, str(i + 1) + "_" + str(i + 2) + ".png"), image_matches)
