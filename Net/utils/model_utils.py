import torch.nn as nn

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