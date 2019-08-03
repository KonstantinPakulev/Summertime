import torch
import torch.nn.functional as F
import numpy
import cv2

from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid, warp_keypoints
from Net.utils.model_utils import sample_descriptors

torch.manual_seed(9)

score = torch.rand((4, 4))

print(score)

print(score.diag())

score -= torch.eye(score.size(0)) * 2

print(score)


