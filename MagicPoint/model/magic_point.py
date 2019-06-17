import torch
import torch.nn as nn

from common.backbone import make_vgg_backbone, make_detector_head


class MagicPoint(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = make_vgg_backbone()
        self.detector = make_detector_head(self.config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.detector(x)

        # TODO output probabilities
        # TODO nms

        return x


