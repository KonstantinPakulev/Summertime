import torch.nn as nn
import torch.nn.functional as F

from Net.utils.model_utils import make_vgg_backbone, make_vgg_detector_head, make_vgg_descriptor_head


class Net(nn.Module):

    def __init__(self, grid_size, descriptor_size):
        super().__init__()
        self.grid_size = grid_size

        self.backbone = make_vgg_backbone()
        self.detector = make_vgg_detector_head(grid_size)
        self.descriptor = make_vgg_descriptor_head(descriptor_size)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """
        x = self.backbone(x)
        raw_score = self.detector(x)
        raw_desc = self.descriptor(x)

        raw_score = raw_score.softmax(dim=1)
        # Remove 'no interest point' channel
        raw_score = raw_score[:, :-1, :, :]
        score = F.pixel_shuffle(raw_score, self.grid_size)

        desc = F.normalize(raw_desc)

        return score, desc
