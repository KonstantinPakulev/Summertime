import torch.nn as nn

from legacy.common.model_utils import make_vgg_backbone, make_detector_head, DepthToSpace


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

        return {'logits': x, 'probs': probs}