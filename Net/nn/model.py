import torch.nn as nn
import torch.nn.functional as F

from Net.utils.model_utils import make_vgg_backbone, make_vgg_descriptor_head


class Net(nn.Module):

    def __init__(self, grid_size, descriptor_size, train_mode=True):
        super().__init__()
        self.grid_size = grid_size

        self.backbone = make_vgg_backbone()
        self.descriptor = make_vgg_descriptor_head(descriptor_size)
        self.train_mode = train_mode

    def forward(self, x):
        """
        :param x: B x C x H x W
        """
        x = self.backbone(x)
        raw_desc = self.descriptor(x)

        if self.train_mode:
            desc = F.normalize(raw_desc)
        else:
            desc = F.interpolate(raw_desc, scale_factor=self.grid_size, mode='bilinear', align_corners=True)
            desc = F.normalize(desc)

        return desc
