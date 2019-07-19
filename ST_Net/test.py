import torch
import torch.nn.functional as F

from ST_Net.model.st_net_vgg import STNetVGGModule

from MagicPoint.dataset.homographies import warp_points_torch
import kornia


tnzr = torch.rand((1, 1, 24, 32))
homo = torch.tensor([[0.7088, -0.010965, -26.07],
                     [-0.13602, 0.83489, 103.19],
                     [-0.00023352, -1.5615e-05, 1.0004]]).unsqueeze(0)

print(STNetVGGModule.get_des_correspondence_mask(tnzr, homo).shape)

