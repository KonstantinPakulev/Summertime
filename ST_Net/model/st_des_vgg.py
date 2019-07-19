import torch
import torch.nn as nn
import torch.nn.functional as F

from ST_Net.utils.model_utils import make_vgg_descriptor_head


class STDesVGGModule(nn.Module):
    def __init__(self,
                 grid_size,
                 descriptor_size):
        super(STDesVGGModule, self).__init__()

        self.grid_size = grid_size
        self.descriptor_size = descriptor_size

        self.descriptor_head = make_vgg_descriptor_head(self.descriptor_size)

    def forward(self, features):
        # Raw descriptor for training
        raw_desc = self.descriptor_head(features)

        desc = F.interpolate(raw_desc, scale_factor=self.grid_size, mode='bilinear', align_corners=True)
        desc = F.normalize(desc)

        return raw_desc, desc

    @staticmethod
    def loss(left_desc, right_desc, c_mask):
        batch_size, c, Hc, Wc = left_desc.size()

        descriptors = left_desc.view([batch_size, Hc, Wc, 1, 1, -1])
        warped_descriptors = right_desc.view([batch_size, 1, 1, Hc, Wc, -1])

        dp_desc = torch.sum(descriptors * warped_descriptors, dim=-1)

        positive_dist = (-dp_desc + 1.0).clamp(min=0.0)
        negative_dist = (dp_desc - 0.2).clamp(min=0.0)

        loss = 800 * c_mask * positive_dist + (
                    torch.tensor(1, dtype=torch.float).to(c_mask.device) - c_mask) * negative_dist
        loss /= Hc * Wc

        return loss.mean()



