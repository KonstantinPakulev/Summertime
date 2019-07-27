import torch.nn as nn
import torch.nn.functional as F

from Net.utils.model_utils import make_vgg_backbone, make_vgg_detector_head, make_vgg_descriptor_head


class Net(nn.Module):

    def __init__(self, grid_size, descriptor_size, train_mode=True):
        super().__init__()
        self.grid_size = grid_size

        self.backbone = make_vgg_backbone()
        self.detector = make_vgg_detector_head(grid_size)
        self.descriptor = make_vgg_descriptor_head(descriptor_size)
        self.train_mode = train_mode

    def forward(self, x):
        """
        :param x: B x C x H x W
        """
        x = self.backbone(x)
        raw_det = self.detector(x)
        raw_des = self.descriptor(x)

        raw_det = raw_det.softmax(dim=1)
        # Remove 'no interest point' channel
        raw_det = raw_det[:, :-1, :, :]
        det = F.pixel_shuffle(raw_det, self.grid_size)

        if self.train_mode:
            des = F.normalize(raw_des)
        else:
            des = F.interpolate(raw_des, scale_factor=self.grid_size, mode='bilinear', align_corners=True)
            des = F.normalize(des)

        return det, des
