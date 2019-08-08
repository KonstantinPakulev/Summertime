import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.utils.model_utils import (make_vgg_ms_block,
                                   make_sdc_ms_block,
                                   make_rf_ms_block,
                                   make_vgg_descriptor,

                                   multi_scale_nms,
                                   multi_scale_softmax)


class Net(nn.Module):

    def __init__(self, descriptor_size):
        super().__init__()

        self.detector = NetDetector()
        self.descriptor = NetDescriptor(descriptor_size)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        score, x = self.detector(x)
        desc = self.descriptor(x)

        return score, desc


class NetDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = make_sdc_ms_block(1, 16, 1)
        self.conv2 = make_sdc_ms_block(1, 16, 2)
        self.conv3 = make_sdc_ms_block(1, 16, 3)
        self.conv4 = make_sdc_ms_block(1, 16, 4)

        self.conv5 = make_sdc_ms_block(64, 16, 1)
        self.conv6 = make_sdc_ms_block(64, 16, 2)
        self.conv7 = make_sdc_ms_block(64, 16, 3)
        self.conv8 = make_sdc_ms_block(64, 16, 4)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        s1 = self.conv1(x)
        s2 = self.conv2(x)
        s3 = self.conv3(x)
        s4 = self.conv4(x)

        x = torch.cat((s1, s2, s3, s4), dim=1)

        s5 = self.conv5(x)
        s6 = self.conv6(x)
        s7 = self.conv7(x)
        s8 = self.conv8(x)

        x = torch.cat((s5, s6, s7, s8), dim=1)

        multi_scale_scores = multi_scale_nms(x, 15)
        score = multi_scale_softmax(multi_scale_scores)

        return score, x


class NetDescriptor(nn.Module):

    def __init__(self, descriptor_size):
        super().__init__()

        self.descriptor = make_vgg_descriptor(descriptor_size)

    def forward(self, x):

        x = self.descriptor(x)
        desc = F.normalize(x)

        return desc


class NetVGG(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1, self.score1 = make_vgg_ms_block(1, 16)
        self.conv2, self.score2 = make_vgg_ms_block(16, 16)

        self.conv3, self.score3 = make_vgg_ms_block(16, 16)
        self.conv4, self.score4 = make_vgg_ms_block(16, 16)

        self.conv5, self.score5 = make_vgg_ms_block(16, 16)
        self.conv6, self.score6 = make_vgg_ms_block(16, 16)

        self.conv7, self.score7 = make_vgg_ms_block(16, 16)
        self.conv8, self.score8 = make_vgg_ms_block(16, 16)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        x = self.conv1(x)
        s1 = self.score1(x)

        x = self.conv2(x)
        s2 = self.score2(x)

        x = self.conv3(x)
        s3 = self.score3(x)

        x = self.conv4(x)
        s4 = self.score4(x)

        x = self.conv5(x)
        s5 = self.score5(x)

        x = self.conv6(x)
        s6 = self.score6(x)

        x = self.conv7(x)
        s7 = self.score7(x)

        x = self.conv8(x)
        s8 = self.score8(x)

        multi_scale_scores = torch.cat((s1, s2, s3, s4, s5, s6, s7, s8), dim=1)

        multi_scale_scores = multi_scale_nms(multi_scale_scores, 15)
        score = multi_scale_softmax(multi_scale_scores)

        return score


class NetRF(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1, self.score1 = make_rf_ms_block(1, 16)
        self.conv2, self.score2 = make_rf_ms_block(16, 16)
        self.conv3, self.score3 = make_rf_ms_block(16, 16)
        self.conv4, self.score4 = make_rf_ms_block(16, 16)
        self.conv5, self.score5 = make_rf_ms_block(16, 16)
        self.conv6, self.score6 = make_rf_ms_block(16, 16)
        self.conv7, self.score7 = make_rf_ms_block(16, 16)
        self.conv8, self.score8 = make_rf_ms_block(16, 16)
        self.conv9, self.score9 = make_rf_ms_block(16, 16)
        self.conv10, self.score10 = make_rf_ms_block(16, 16)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        x = self.conv1(x)
        s1 = self.score1(x)

        x = self.conv2(x)
        s2 = self.score2(x)

        x = self.conv3(x)
        s3 = self.score3(x)

        x = self.conv4(x)
        s4 = self.score4(x)

        x = self.conv5(x)
        s5 = self.score5(x)

        x = self.conv6(x)
        s6 = self.score6(x)

        x = self.conv7(x)
        s7 = self.score7(x)

        x = self.conv8(x)
        s8 = self.score8(x)

        x = self.conv9(x)
        s9 = self.score9(x)

        x = self.conv10(x)
        s10 = self.score10(x)

        multi_scale_scores = torch.cat((s1, s2, s3, s4, s5, s6, s7, s8, s9, s10), dim=1)

        multi_scale_scores = multi_scale_nms(multi_scale_scores, 15)
        score = multi_scale_softmax(multi_scale_scores)

        return score
