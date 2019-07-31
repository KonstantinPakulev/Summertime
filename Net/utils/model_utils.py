import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Building blocks for models
"""


def make_rf_block(in_channels, out_channels):
    conv = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    conv += [nn.InstanceNorm2d(out_channels, affine=True)]
    conv += [nn.LeakyReLU()]

    score = [nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)]
    score += [nn.InstanceNorm2d(1, affine=True)]

    return nn.Sequential(*conv), nn.Sequential(*score)


def make_vgg_block(in_channels, out_channels):
    conv = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    conv += [nn.BatchNorm2d(out_channels)]
    conv += [nn.ReLU()]

    score = [nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)]
    score += [nn.BatchNorm2d(1)]

    return nn.Sequential(*conv), nn.Sequential(*score)


def make_sdc_block(in_channels, out_channels, dilation):
    conv = [nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2 + 2 * (dilation - 1), dilation=dilation)]
    conv += [nn.ELU()]

    return nn.Sequential(*conv)


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


def multi_scale_nms(multi_scale_scores, k_size, strength=3.0):
    padding = k_size // 2

    nms_scale_scores = F.max_pool2d(multi_scale_scores, kernel_size=k_size, padding=padding, stride=1)
    max_scale_scores, _ = nms_scale_scores.max(dim=1)

    _, c, _, _ = multi_scale_scores.size()

    exp = torch.exp(strength * (multi_scale_scores - max_scale_scores))
    weight = torch.ones((1, c, k_size, k_size)).to(multi_scale_scores.device)
    sum_exp = F.conv2d(exp, weight=weight, padding=padding) + 1e-8

    return exp / sum_exp


def multi_scale_softmax(multi_scale_scores, strength=100.0):
    max_scores, _ = multi_scale_scores.max(dim=1, keepdim=True)

    exp = torch.exp(strength * (multi_scale_scores - max_scores))
    sum_exp = exp.sum(dim=1, keepdim=True) + 1e-8
    softmax = exp / sum_exp

    score = torch.sum(multi_scale_scores * softmax, dim=1, keepdim=True)

    return score


def sample_descriptors(desc, kp, grid_size):
    """
    :param desc: N x C x H x W
    :param kp: N x 4
    :param grid_size: int
    """
    desc = F.interpolate(desc, scale_factor=grid_size, mode='bilinear', align_corners=True)
    desc = F.normalize(desc)

    return desc[kp[:, 0], :, kp[:, 2], kp[:, 3]]
