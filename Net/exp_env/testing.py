import torch
import torch.nn.functional as F
import numpy
import cv2
from bisect import bisect

from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid, warp_keypoints, gaussian_filter
from Net.utils.model_utils import sample_descriptors
from Net.utils.math_utils import calculate_similarity_matrix, calculate_similarity_vector

from Net.nn.criterion import HardTripletLoss

torch.set_printoptions(precision=3, sci_mode=False, linewidth=9099999)

t1 = torch.rand((512, 32))
t2 = torch.rand((512, 32))

t1 /= torch.norm(t1, dim=-1).unsqueeze(1)
t2 /= torch.norm(t2, dim=-1).unsqueeze(1)


# print(calculate_similarity_matrix(t1, t2).diag())
# print(calculate_similarity_vector(t1, t2))