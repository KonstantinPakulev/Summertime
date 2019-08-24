import torch
import torch.nn.functional as F

from Net.utils.common_utils import *

from Net.source.nn.main_criterion import HardTripletLoss, MSELossRF

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
