# import torch
# import torch.nn as nn
#
# from Net.source.utils.math_utils import to_homogeneous, robust_symmetric_epipolar_distance
#
#
# class SymmetricEpipolarDistanceLoss(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, o_kp1, o_nn_kp2, match_mask, F_estimates, norm_transform1, norm_transform2):
#         loss = torch.tensor(0.0).to(o_kp1.device)
#
#         kp1_norm = torch.bmm(to_homogeneous(o_kp1), norm_transform1.permute(0, 2, 1))
#         kp2_norm = torch.bmm(to_homogeneous(o_nn_kp2), norm_transform2.permute(0, 2, 1))
#
#         for F_estimate in F_estimates:
#             loss += (robust_symmetric_epipolar_distance(kp1_norm, kp2_norm, F_estimate) * match_mask.float()).sum() / match_mask.sum().float()
#
#         return loss
