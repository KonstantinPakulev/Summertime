"""
SDC Net
"""


def make_sdc_ms_block(in_channels, out_channels, dilation):
    conv = [nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2 + 2 * (dilation - 1), dilation=dilation)]
    conv += [nn.ELU()]

    return nn.Sequential(*conv)


def make_sdc_score_block(in_channels):
    score = [nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)]
    score += [nn.BatchNorm2d(1)]

    return nn.Sequential(*score)


def make_sdc_descriptor(descriptor_size):
    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += make_vgg_block(64, 64)
    layers += make_vgg_block(64, 64)

    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += make_vgg_block(64, 128)
    layers += make_vgg_block(128, 128)

    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += make_vgg_block(128, 128)
    layers += make_vgg_block(128, 128)

    layers += make_vgg_block(128, 256)
    layers += [nn.Conv2d(256, descriptor_size, kernel_size=1)]

    return nn.Sequential(*layers)



"""
RF Detector
"""


def make_rf_ms_block(in_channels, out_channels):
    conv = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    conv += [nn.InstanceNorm2d(out_channels, affine=True)]
    conv += [nn.LeakyReLU()]

    score = [nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)]
    score += [nn.InstanceNorm2d(1, affine=True)]

    return nn.Sequential(*conv), nn.Sequential(*score)