import torch
import torch.nn as nn
from common.model_utils import make_vgg_backbone, make_detector_head, DepthToSpace


class MagicPoint(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = make_vgg_backbone()
        self.detector = make_detector_head(self.config)
        self.detector_upscale = DepthToSpace(self.config['grid_size'])

    def forward(self, x):
        x = self.backbone(x)
        x = self.detector(x)

        probs = x.softmax(dim=1)
        # Remove the last bin, since it stands for "no interest point"
        probs = probs[:, :-1, :, :]
        probs = self.detector_upscale(probs)
        probs = probs.squeeze()

        # TODO. Homography adaptation?
        # TODO. Non-maximum supression.

        # if self.config['nms']: for p in probs: box_nms(p, self.config['nms'], min_prob=self.config[
        # 'detection_threshold'], keep_top_k=self.config['top_k'])

        return {'logits': x, 'probs': probs}
