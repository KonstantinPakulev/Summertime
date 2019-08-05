import torch
import torch.nn.functional as F
import numpy
import cv2
from bisect import bisect

from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid, warp_keypoints, gaussian_filter
from Net.utils.model_utils import sample_descriptors
from Net.nn.criterion import ReceptiveHingeLoss, HardTripletLoss

torch.set_printoptions(precision=3, sci_mode=False, linewidth=9099999)

t = torch.zeros((1, 1, 30, 40))
t[0, 0, 15, 20] = 1
t1 = gaussian_filter(t, 31, 7)
t2 = gaussian_filter(t, 11, 4)
t3 = gaussian_filter(t, 5, 2)

print(t1.sum() + t2.sum() + t3.sum())

print(t)

