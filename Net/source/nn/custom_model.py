import torch
import torch.nn as nn
import torch.nn.functional as F

from Net.source.utils.model_utils import (make_rf_ms_block,
                                          make_sdc_ms_block,
                                          make_sdc_score_block,

                                          make_sdc_descriptor,

                                          multi_scale_nms,
                                          multi_scale_softmax)


class DetectorRF(nn.Module):

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


class NetSDC(nn.Module):

    def __init__(self, descriptor_size):
        super().__init__()

        self.detector = NetSDCDetector()
        self.descriptor = NetSDCDescriptor(descriptor_size)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        score, x = self.detector(x)
        desc = self.descriptor(x)

        return score, None


class NetSDCDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = make_sdc_ms_block(1, 16, 1)
        self.conv2 = make_sdc_ms_block(1, 16, 2)
        self.conv3 = make_sdc_ms_block(1, 16, 3)
        self.conv4 = make_sdc_ms_block(1, 16, 4)

        self.score1 = make_sdc_score_block(64)

        self.conv5 = make_sdc_ms_block(64, 32, 1)
        self.conv6 = make_sdc_ms_block(64, 32, 2)
        self.conv7 = make_sdc_ms_block(64, 32, 3)
        self.conv8 = make_sdc_ms_block(64, 32, 4)

        self.score2 = make_sdc_score_block(128)

        self.conv9 = make_sdc_ms_block(128, 32, 1)
        self.conv10 = make_sdc_ms_block(128, 32, 2)
        self.conv11 = make_sdc_ms_block(128, 32, 4)
        self.conv12 = make_sdc_ms_block(128, 32, 3)

        self.score3 = make_sdc_score_block(128)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """

        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)

        x = torch.cat((c1, c2, c3, c4), dim=1)

        s1 = self.score1(x)

        c5 = self.conv5(x)
        c6 = self.conv6(x)
        c7 = self.conv7(x)
        c8 = self.conv8(x)

        x = torch.cat((c5, c6, c7, c8), dim=1)

        s2 = self.score2(x)

        c9 = self.conv9(x)
        c10 = self.conv10(x)
        c11 = self.conv11(x)
        c12 = self.conv12(x)

        x = torch.cat((c9, c10, c11, c12), dim=1)

        s3 = self.score3(x)

        multi_scale_scores = multi_scale_nms(torch.cat((s1, s2, s3), dim=1), 15)
        score = multi_scale_softmax(multi_scale_scores)

        return score, None


class NetSDCDescriptor(nn.Module):

    def __init__(self, descriptor_size):
        super().__init__()

        self.descriptor = make_sdc_descriptor(descriptor_size)

    def forward(self, x):

        x = self.descriptor(x)
        desc = F.normalize(x)

        return desc
