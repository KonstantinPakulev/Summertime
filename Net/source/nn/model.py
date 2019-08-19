import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.source.utils.model_utils import (make_vgg_ms_block,
                                          make_vgg_ms_detector,
                                          make_vgg_ms_descriptor,

                                          multi_scale_nms,
                                          multi_scale_softmax,
                                          multi_scale_nms_softmax)


class NetVGG(nn.Module):

    def __init__(self, grid_size, descriptor_size):
        super().__init__()

        self.grid_size = grid_size
        self.descriptor_size = descriptor_size

        self.conv1, self.score1 = make_vgg_ms_block(1, 64, 1)
        self.conv2, self.score2 = make_vgg_ms_block(64, 64, 1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3, self.score3 = make_vgg_ms_block(64, 64, 2)
        self.conv4, self.score4 = make_vgg_ms_block(64, 64, 2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5, self.score5 = make_vgg_ms_block(64, 128, 4)
        self.conv6, self.score6 = make_vgg_ms_block(128, 128, 4)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7, self.score7 = make_vgg_ms_block(128, 128, 8)
        self.conv8, self.score8 = make_vgg_ms_block(128, 128, 8)

        self.detector = make_vgg_ms_detector(128, 256, self.grid_size)
        self.descriptor = make_vgg_ms_descriptor(128, 256, self.descriptor_size)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        x = self.conv1(x)
        s1 = self.score1(x)

        x = self.conv2(x)
        s2 = self.score2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        s3 = self.score3(x)

        x = self.conv4(x)
        s4 = self.score4(x)

        x = self.pool2(x)

        x = self.conv5(x)
        s5 = self.score5(x)

        x = self.conv6(x)
        s6 = self.score6(x)

        x = self.pool3(x)

        x = self.conv7(x)
        s7 = self.score7(x)

        x = self.conv8(x)
        s8 = self.score8(x)

        s9 = self.detector(x)

        multi_scale_scores = torch.cat((s1, s2, s3, s4, s5, s6, s7, s8, s9), dim=1)
        multi_scale_scores = multi_scale_nms(multi_scale_scores, 15)
        score = multi_scale_softmax(multi_scale_scores)

        desc = self.descriptor(x)
        desc = F.normalize(desc)

        return score, desc


class NetVGGDebug(NetVGG):

    def __init__(self, grid_size, descriptor_size):
        super().__init__(grid_size, descriptor_size)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        x = self.conv1(x)
        s1 = self.score1(x)

        x = self.conv2(x)
        s2 = self.score2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        s3 = self.score3(x)

        x = self.conv4(x)
        s4 = self.score4(x)

        x = self.pool2(x)

        x = self.conv5(x)
        s5 = self.score5(x)

        x = self.conv6(x)
        s6 = self.score6(x)

        x = self.pool3(x)

        x = self.conv7(x)
        s7 = self.score7(x)

        x = self.conv8(x)
        s8 = self.score8(x)

        s9 = self.detector(x)

        multi_scale_scores = torch.cat((s1, s2, s3, s4, s5, s6, s7, s8, s9), dim=1)
        # multi_scale_scores_nms = multi_scale_nms(multi_scale_scores, 15)
        # score = multi_scale_softmax(multi_scale_scores_nms)
        score = multi_scale_nms_softmax(multi_scale_scores)

        desc = self.descriptor(x)
        desc = F.normalize(desc)

        return score, desc, {
            'mss': multi_scale_scores,
            # 'mss_nms': multi_scale_scores_nms,
        }

