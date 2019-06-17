import torch.nn as nn

vgg_structure = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128]


def make_vgg_block(in_channels, out_channels, kernel_size, activation):
    block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)]
    block += nn.BatchNorm2d(out_channels)

    if activation is not None:
        block += activation

    return block


def make_vgg_backbone():
    layers = []
    in_channels = 3
    for v in vgg_structure:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += make_vgg_block(in_channels, v, 3, nn.ReLU(inplace=True))
            in_channels = v

    return nn.Sequential(*layers)


def make_detector_head(config):
    layers = [make_vgg_block(vgg_structure[-1], 256, 3, nn.ReLU(inplace=True)),
              make_vgg_block(256, 1 + pow(config['grid_size'], 2), 1, None)]
    layers += nn.Softmax(dim=1)

    return nn.Sequential(*layers)





