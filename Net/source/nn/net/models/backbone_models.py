import torch.nn as nn

import Net.source.core.experiment as exp

from Net.source.nn.net.utils.model_utils import vgg_block


class VGGJointBackbone(nn.Module):

    @staticmethod
    def from_config(model_config):
        input_channels = model_config[exp.INPUT_CHANNELS]
        batch_norm = model_config[exp.BATCH_NORM]
        return VGGJointBackbone(input_channels, batch_norm)

    def __init__(self, input_channels, batch_norm):
        super().__init__()
        self.conv1 = vgg_block(input_channels, 64, batch_norm=batch_norm)
        self.conv2 = vgg_block(64, 64, batch_norm=batch_norm)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = vgg_block(64, 64, batch_norm=batch_norm)
        self.conv4 = vgg_block(64, 64, batch_norm=batch_norm)

        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = vgg_block(64, 128, batch_norm=batch_norm)
        self.conv6 = vgg_block(128, 128, batch_norm=batch_norm)

        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = vgg_block(128, 128, batch_norm=batch_norm)
        self.conv8 = vgg_block(128, 128, batch_norm=batch_norm)

    def forward(self, image):
        x1 = self.conv1(image)
        x2 = self.conv2(x1)

        x = self.max_pool1(x2)

        x = self.conv3(x)
        x4 = self.conv4(x)

        x = self.max_pool2(x4)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.max_pool3(x)

        x = self.conv7(x)
        x = self.conv8(x)

        return [x, x1, x2, x4]


# Legacy backbones
#
#
# class VGGMSDBackbone(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1, self.score1 = make_vgg_ms_block(1, 64, 1)
#         self.conv2, self.score2 = make_vgg_ms_block(64, 64, 1)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3, self.score3 = make_vgg_ms_block(64, 64, 2)
#         self.conv4, self.score4 = make_vgg_ms_block(64, 64, 2)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv5, self.score5 = make_vgg_ms_block(64, 128, 4)
#         self.conv6, self.score6 = make_vgg_ms_block(128, 128, 4)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv7, self.score7 = make_vgg_ms_block(128, 128, 8)
#         self.conv8, self.score8 = make_vgg_ms_block(128, 128, 8)
#
#     def forward(self, image):
#         x = self.conv1(image)
#         s1 = self.score1(x)
#
#         x = self.conv2(x)
#         s2 = self.score2(x)
#
#         x = self.pool1(x)
#
#         x = self.conv3(x)
#         s3 = self.score3(x)
#
#         x = self.conv4(x)
#         s4 = self.score4(x)
#
#         x = self.pool2(x)
#
#         x = self.conv5(x)
#         s5 = self.score5(x)
#
#         x = self.conv6(x)
#         s6 = self.score6(x)
#
#         x = self.pool3(x)
#
#         x = self.conv7(x)
#         s7 = self.score7(x)
#
#         x = self.conv8(x)
#         s8 = self.score8(x)
#
#         return [x, s1, s2, s3, s4, s5, s6, s7, s8]
#
#
# class VGGMSBackbone(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1 = make_vgg_block(1, 64)
#         self.conv2, self.score1 = make_vgg_ms_block(64, 64, 1)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = make_vgg_block(64, 64)
#         self.conv4, self.score2 = make_vgg_ms_block(64, 64, 2)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv5 = make_vgg_block(64, 128)
#         self.conv6, self.score3 = make_vgg_ms_block(128, 128, 4)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv7 = make_vgg_block(128, 128)
#         self.conv8, self.score4 = make_vgg_ms_block(128, 128, 8)
#
#     def forward(self, image):
#         x = self.conv1(image)
#         x = self.conv2(x)
#
#         s1 = self.score1(x)
#
#         x = self.pool1(x)
#
#         x = self.conv3(x)
#         x = self.conv4(x)
#
#         s2 = self.score2(x)
#
#         x = self.pool2(x)
#
#         x = self.conv5(x)
#         x = self.conv6(x)
#
#         s3 = self.score3(x)
#
#         x = self.pool3(x)
#
#         x = self.conv7(x)
#         x = self.conv8(x)
#
#         s4 = self.score4(x)
#
#         return [x, s1, s2, s3, s4]
#
#
# class VGGJointOldBackbone(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1 = make_vgg_block(1, 64)
#         self.conv2 = make_vgg_block(64, 64)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = make_vgg_block(64, 64)
#         self.conv4 = make_vgg_block(64, 64)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv5 = make_vgg_block(64, 128)
#         self.conv6 = make_vgg_block(128, 128)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv7 = make_vgg_block(128, 128)
#         self.conv8 = make_vgg_block(128, 128)
#
#     def forward(self, image):
#         x = self.conv1(image)
#         x2 = self.conv2(x)
#
#         x = self.pool1(x2)
#
#         x = self.conv3(x)
#         x4 = self.conv4(x)
#
#         x = self.pool2(x4)
#
#         x = self.conv5(x)
#         x = self.conv6(x)
#
#         x = self.pool3(x)
#
#         x = self.conv7(x)
#         x = self.conv8(x)
#
#         return [x, x2, x4]
