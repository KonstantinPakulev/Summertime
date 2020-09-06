import torch
import torch.nn as nn

import Net.source.core.experiment as exp

from Net.source.nn.net.utils.model_utils import vgg_block


class LocBranch(nn.Module):

    @staticmethod
    def from_config(model_config):
        return LocBranch(model_config[exp.BATCH_NORM])

    def __init__(self, batch_norm):
        super().__init__()
        self.loc_conv1 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.loc_conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        self.loc_conv3 = vgg_block(4, 4, kernel_size=3, batch_norm=batch_norm)
        self.loc_conv4 = vgg_block(4, 4, kernel_size=3, batch_norm=batch_norm)
        self.loc_conv5 = nn.Conv2d(4, 2, kernel_size=1, padding=0)

    def forward(self, x1, x2):
        x1 = self.loc_conv1(x1)
        x2 = self.loc_conv2(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self.loc_conv3(x)
        x = self.loc_conv4(x)
        loc = self.loc_conv5(x).clamp(min=-5.0, max=5.0)

        return loc


# # Legacy models
#
# class LocOldBranch(nn.Module):
#
#     @staticmethod
#     def from_config(model_config):
#         return LocOldBranch(model_config[exp.NUM_LOC_SCORES],
#                             model_config[exp.LOC_KERNEL_SIZE])
#
#     def __init__(self, num_loc_scores, loc_kernel_size):
#         super().__init__()
#         self.loc_range = float(loc_kernel_size) / 2
#
#         self.loc = nn.Sequential(make_vgg_block(num_loc_scores, 2, kernel_size=loc_kernel_size),
#                                  make_sal_detector_block(2, 2, module=True))
#
#     def forward(self, ms_log_score):
#         loc = self.loc(ms_log_score)
#         loc = loc.clamp(min=-self.loc_range, max=self.loc_range)
#
#         return loc
