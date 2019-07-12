import torch.nn as nn
import torch.nn.functional as F

from common.model_utils import make_vgg_backbone, make_detector_head, make_descriptor_head, DepthToSpace


class SuperPoint(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = make_vgg_backbone()

        self.detector = make_detector_head(self.config)
        self.detector_upscale = DepthToSpace(self.config['grid_size'])

        self.descriptor = make_descriptor_head(self.config)

    def forward(self, x):
        x = self.backbone(x)

        x_det = self.detector(x)
        x_des = self.descriptor(x)

        probs = x_det.softmax(dim=1)
        # Remove the last bin, since it stands for "no interest point"
        probs = probs[:, :-1, :, :]
        probs = self.detector_upscale(probs)
        probs = probs.squeeze()

        descs = F.interpolate(x_des, scale_factor=self.config['grid_size'], mode='bilinear', align_corners=True)
        descs = F.normalize(descs)

        return {'logits': x_det, 'probs': probs, 'raw_desc': x_des, 'desc': descs}


