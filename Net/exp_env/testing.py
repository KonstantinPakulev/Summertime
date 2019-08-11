import torch
import torch.nn.functional as F
import numpy
import cv2
from bisect import bisect

from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid, \
    warp_keypoints, gaussian_filter, select_keypoints, erode_filter
from Net.utils.model_utils import sample_descriptors
from Net.utils.common_utils import *
from Net.utils.math_utils import calculate_inv_similarity_matrix, calculate_inv_similarity_vector

from Net.nn.criterion import HardTripletLoss, MSELoss

torch.set_printoptions(precision=3, sci_mode=False, linewidth=9099999)

t = torch.tensor([[[3, 2], [4, 3], [5, 4]],
                  [[5, 4], [6, 5], [7, 6]]])

i = torch.tensor([[1, 0, 0],
                  [1, 1, 0]]).unsqueeze(-1)

print(t * i)

# i = torch.tensor([[2, 1, 0],
#                   [2, 1, 0],
#                   [2, 1, 0]])
#
# print(t.shape)
# print(i.shape)
# id = torch.tensor([0, 1]).unsqueeze(0).unsqueeze(0).repeat(3, 3, 1)
# print(id.shape)

# print(i.shape)
# i = torch.cat((i, id), dim=-1)
#
# print(i.shape)
# print(i)
