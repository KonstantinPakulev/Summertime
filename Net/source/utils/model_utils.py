import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Building blocks for models
"""


def make_vgg_block(in_channels, out_channels):
    conv = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    conv += [nn.BatchNorm2d(out_channels)]
    conv += [nn.ReLU()]

    return conv


"""
VGG Net
"""


def make_vgg_ms_block(in_channels, out_channels, factor):
    conv = make_vgg_block(in_channels, out_channels)

    score = [nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)]
    score += [nn.UpsamplingBilinear2d(scale_factor=factor)]
    score += [nn.BatchNorm2d(1)]

    return nn.Sequential(*conv), nn.Sequential(*score)


def make_vgg_ms_detector(in_channels, out_channels, grid_size):
    conv = make_vgg_block(in_channels, out_channels)
    conv += [nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)]
    conv += [nn.UpsamplingBilinear2d(scale_factor=grid_size)]
    conv += [nn.BatchNorm2d(1)]

    return nn.Sequential(*conv)


def make_vgg_ms_descriptor(in_channels, out_channels, descriptor_size):
    conv = make_vgg_block(in_channels, out_channels)
    conv += [nn.Conv2d(out_channels, descriptor_size, kernel_size=1, padding=0)]

    return nn.Sequential(*conv)


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
Tensor functions
"""


def multi_scale_nms(multi_scale_scores, k_size, strength=3.0):
    padding = k_size // 2

    nms_score = F.max_pool2d(multi_scale_scores, kernel_size=k_size, padding=padding, stride=1)
    max_nms_score, _ = nms_score.max(dim=1)

    exp = torch.exp(strength * (multi_scale_scores - max_nms_score.unsqueeze(1)))
    weight = torch.ones((1, multi_scale_scores.size(1), k_size, k_size)).to(multi_scale_scores.device)
    sum_exp = F.conv2d(exp, weight=weight, padding=padding) + 1e-8

    return exp / sum_exp


def multi_scale_softmax(multi_scale_scores, strength=100.0):
    max_score, _ = multi_scale_scores.max(dim=1, keepdim=True)

    exp = torch.exp(strength * (multi_scale_scores - max_score))
    sum_exp = exp.sum(dim=1, keepdim=True) + 1e-8
    softmax = exp / sum_exp

    score = torch.sum(multi_scale_scores * softmax, dim=1, keepdim=True)

    return score


def sample_descriptors(desc, kp, grid_size):
    """
    :param desc: B x C x H x W
    :param kp: B x N x 2
    :param grid_size: int
    :return B x N x C
    """
    _, _, h, w = desc.size()

    kp_grid = kp[:, :, [1, 0]].float().unsqueeze(1) / grid_size
    kp_grid[:, :, :, 0] = kp_grid[:, :, :, 0] / (w - 1) * 2 - 1
    kp_grid[:, :, :, 1] = kp_grid[:, :, :, 1] / (h - 1) * 2 - 1

    return F.normalize(F.grid_sample(desc, kp_grid).squeeze(2)).permute(0, 2, 1)


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
