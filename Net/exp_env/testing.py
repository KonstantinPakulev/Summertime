import torch
import torch.nn.functional as F
import numpy
import cv2
from bisect import bisect

from Net.utils.image_utils import create_coordinates_grid, warp_coordinates_grid, warp_keypoints
from Net.utils.model_utils import sample_descriptors
from Net.nn.criterion import ReceptiveHomoHingeLoss

dkss = [11, 9]
dksi = [5, 7]

inddex= bisect(dksi, 8)
if inddex >= len(dksi):
    inddex = len(dksi) - 1

print(inddex)

# loss = ReceptiveHomoHingeLoss(8, 1, 0.2, 1)
#
# torch.manual_seed(9)
#
# kp1 = torch.rand((512, 4))
# desc1 = torch.rand((1, 4, 30, 40))
# desc2 = torch.rand((1, 4, 30, 40))
# homo21 = torch.rand((1, 3, 3))
# vis_mask1 = torch.rand((1, 1, 240, 320))
#
# print(loss(kp1, desc1 ,desc2, homo21, vis_mask1))

