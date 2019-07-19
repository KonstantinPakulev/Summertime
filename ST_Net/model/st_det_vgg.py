import torch
import torch.nn as nn
import torch.nn.functional as F

from ST_Net.utils.model_utils import make_vgg_detector_head
from ST_Net.utils.image_utils import filter_border, nms, top_k_map, get_gauss_filter_weight


class STDetVGGModule(nn.Module):

    def __init__(self,
                 grid_size,
                 nms_thresh,
                 nms_k_size,
                 top_k,
                 gauss_k_size,
                 gauss_sigma
                 ):
        super(STDetVGGModule, self).__init__()

        self.grid_size = grid_size
        self.nms_thresh = nms_thresh
        self.nms_k_size = nms_k_size
        self.top_k = top_k
        self.gauss_k_size = gauss_k_size
        self.gauss_sigma = gauss_sigma

        self.detector_head = make_vgg_detector_head(self.grid_size)

    def forward(self, features):
        """
        :param features: Abstract features from VGG backbone of shape (N, C, H/8, W/8)
        """

        x = self.detector_head(features)  # (N, 1 + grid_size^2, H/8, W/8)
        x = x.softmax(dim=1)  # Convert logits to probabilities
        x = x[:, :-1, :, :]  # Last bin stands for "no interest point", so we remove it
        score_map = F.pixel_shuffle(x, self.grid_size)  # (N, 1, H, W)

        return score_map

    def process(self, w_score_map):
        """
        :param w_score_map: warped score map of size # (N, 1, H, W)
        """

        w_score_map = filter_border(w_score_map)

        nms_mask = nms(w_score_map, thresh=self.nms_thresh, k_size=self.nms_k_size)
        w_score_map = w_score_map * nms_mask
        top_k_value = w_score_map

        # Select top k values from w_score_map
        top_k_mask = top_k_map(w_score_map, self.top_k)
        w_score_map = top_k_mask.to(torch.float) * w_score_map

        # Create a clean score map by placing gaussian kernels
        psf = w_score_map.new_tensor(
            get_gauss_filter_weight(self.gauss_k_size, self.gauss_sigma)[None, None, :, :]
        )
        w_score_map = F.conv2d(
            input=w_score_map,
            weight=psf,
            stride=1,
            padding=self.gauss_k_size // 2,
        )

        w_score_map = w_score_map.clamp(min=0.0, max=1.0)

        return w_score_map, top_k_mask, top_k_value

    @staticmethod
    def loss(left_score, ground_truth, gt_visible_mask):
        """
        :param left_score: Left image score
        :param ground_truth: Ground truth for left image score warped from right image score
        :param gt_visible_mask: Visibility mask. Those parts that are present on the right image and on the left
        :return:
        """
        mse = (left_score - ground_truth) ** 2

        num_visible = torch.clamp(gt_visible_mask.sum(dim=(1, 3, 2)), min=2.0)
        loss = (
                torch.sum(mse * gt_visible_mask, dim=(1, 3, 2)) / (num_visible + 1e-8)
        ).mean()

        return loss



