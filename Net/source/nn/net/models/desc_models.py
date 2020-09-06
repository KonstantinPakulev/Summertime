import torch.nn as nn

from torch.nn.functional import normalize

import Net.source.core.experiment as exp

from Net.source.nn.net.utils.model_utils import vgg_block


class DescBranch(nn.Module):

    @staticmethod
    def from_config(model_config):
        return DescBranch(model_config[exp.DESCRIPTOR_SIZE],
                          model_config[exp.BATCH_NORM])

    def __init__(self, descriptor_size, batch_norm):
        super().__init__()
        self.desc_conv1 = vgg_block(128, 256, batch_norm=batch_norm)
        self.desc_conv2 = nn.Conv2d(256, descriptor_size, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.desc_conv1(x)
        desc = self.desc_conv2(x)
        desc = normalize(desc)

        return desc


# Legacy models
#
# class DescOldBranch(nn.Module):
#
#     @staticmethod
#     def from_config(model_config):
#         return DescOldBranch(model_config[exp.DESCRIPTOR_SIZE])
#
#     def __init__(self, descriptor_size):
#         super().__init__()
#
#         self.descriptor = make_descriptor_block(128, 256, descriptor_size)
#
#     def forward(self, x):
#         desc = self.descriptor(x)
#         desc = normalize(desc)
#
#         return desc
