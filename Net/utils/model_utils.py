import torch.nn as nn
import torch.nn.functional as F

"""
VGG based backbone, detector and descriptor
"""

vgg_structure = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128]


def make_vgg_block(in_channels, out_channels, kernel_size, padding, activation):
    block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
    block += [nn.BatchNorm2d(out_channels)]

    if activation is not None:
        block += [activation]

    return block


def make_vgg_backbone():
    layers = []
    in_channels = 1
    for v in vgg_structure:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += make_vgg_block(in_channels, v, 3, 1, nn.ReLU(inplace=True))
            in_channels = v

    return nn.Sequential(*layers)


def make_vgg_detector_head(grid_size):
    layers = []
    layers += make_vgg_block(vgg_structure[-1], 256, 3, 1, nn.ReLU(inplace=True))
    layers += make_vgg_block(256, 1 + pow(grid_size, 2), 1, 0, None)

    return nn.Sequential(*layers)


def make_vgg_descriptor_head(descriptor_size):
    layers = []
    layers += make_vgg_block(vgg_structure[-1], 256, 3, 1, nn.ReLU(inplace=True))
    layers += make_vgg_block(256, descriptor_size, 1, 0, None)

    return nn.Sequential(*layers)


"""
Tensor functions
"""


def space_to_depth(tensor, grid_size):
    """
    :param tensor: N x C x H x W
    :param grid_size: int
    """
    n, c, h, w = tensor.size()

    hr = h // grid_size
    wr = w // grid_size

    x = tensor.view(n, c, hr, grid_size, wr, grid_size)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # N x grid_size x grid_size x C x Hr x Wr
    x = x.view(n, c * (grid_size ** 2), hr, wr)  # N x C * grid_size^2 x Hr x Wr

    return x


def sample_descriptors(desc, kp, grid_size):
    """
    :param desc: N x C x H x W
    :param kp: N x 4
    :param grid_size: int
    """
    desc = F.interpolate(desc, scale_factor=grid_size, mode='bilinear', align_corners=True)
    desc = F.normalize(desc)

    return  desc[kp[:, 0], :, kp[:, 2], kp[:, 3]]
