import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Model info keys
"""
FORWARD_TIME = 'forward_time'
LOG_MS_SCORE = 'ms_log_score'
NMS_MS_SCORE = 'ms_nms_score'

"""
Building blocks for models
"""


def vgg_block(in_channels, out_channels, kernel_size=3, batch_norm=True, module=True):
    padding = (kernel_size - 1) // 2
    conv = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]

    if batch_norm:
        conv += [nn.BatchNorm2d(out_channels)]

    conv += [nn.ReLU()]

    if module:
        return nn.Sequential(*conv)
    else:
        return conv

"""
Score processing functions
"""


def spatial_soft_nms(log_score, soft_nms_kernel_size):
    """
    :param log_score: B x 1 x H x W
    :param soft_nms_kernel_size: int
    :param soft_strength: float
    """
    padding = soft_nms_kernel_size // 2

    max_log_score = F.max_pool2d(log_score, kernel_size=soft_nms_kernel_size, padding=padding,
                                 stride=1)  # B x 1 x H x W

    exp = torch.exp(log_score - max_log_score)
    weight = torch.ones((1, 1, soft_nms_kernel_size, soft_nms_kernel_size)).to(log_score.device)
    sum_exp = F.conv2d(exp, weight=weight, padding=padding).clamp(min=1e-8)

    return exp / sum_exp


# TODO. Need a more sophisticated method of normalization; And think about better name
def normalize_conf_score(conf_score):
    b = conf_score.size(0)

    conf_score = conf_score - conf_score.view(b, -1).min(dim=-1)[0].view(b, 1, 1, 1)
    conf_score = conf_score / conf_score.view(b, -1).max(dim=-1)[0].view(b, 1, 1, 1)
    conf_score = F.interpolate(conf_score, scale_factor=4.0, mode='bilinear').detach()

    return conf_score


# Legacy building blocks


# def make_vgg_block(in_channels, out_channels, kernel_size=3, module=True):
#     padding = (kernel_size - 1) // 2
#     conv = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
#     conv += [nn.BatchNorm2d(out_channels)]
#     conv += [nn.ReLU()]
#
#     if module:
#         return nn.Sequential(*conv)
#     else:
#         return conv
#
#
# def make_sal_detector_block(in_channels, out_channels=1, module=False):
#     det_sal = [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)]
#     det_sal += [nn.BatchNorm2d(out_channels)]
#
#     if module:
#         return nn.Sequential(*det_sal)
#     else:
#         return det_sal
#
#
# def make_vgg_ms_block(in_channels, out_channels, factor):
#     conv = make_vgg_block(in_channels, out_channels)
#
#     score = [nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)]
#     score += [nn.UpsamplingBilinear2d(scale_factor=factor)]
#     score += [nn.BatchNorm2d(1)]
#
#     return conv, nn.Sequential(*score)
#
#
# def make_ms_detector(in_channels, out_channels, grid_size):
#     conv = make_vgg_block(in_channels, out_channels, module=False)
#
#     conv += [nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)]
#     conv += [nn.UpsamplingBilinear2d(scale_factor=grid_size)]
#     conv += [nn.BatchNorm2d(1)]
#
#     return nn.Sequential(*conv)
#
#
# def make_conf_detector_block(in_channels, out_channels):
#     conv = make_vgg_block(in_channels, out_channels, module=False)
#     conv += make_sal_detector_block(out_channels)
#
#     return nn.Sequential(*conv)
#
#
# def make_descriptor_block(in_channels, out_channels, descriptor_size):
#     conv = make_vgg_block(in_channels, out_channels, module=False)
#     conv += [nn.Conv2d(out_channels, descriptor_size, kernel_size=1, padding=0)]
#
#     return nn.Sequential(*conv)
#
# def multi_scale_nms(multi_scale_scores, k_size, strength=3.0):
#     padding = k_size // 2
#
#     nms_score = F.max_pool2d(multi_scale_scores, kernel_size=k_size, padding=padding, stride=1)
#     max_nms_score, _ = nms_score.max(dim=1)
#
#     exp = torch.exp(strength * (multi_scale_scores - max_nms_score.unsqueeze(1)))
#     weight = torch.ones((1, multi_scale_scores.size(1), k_size, k_size)).to(multi_scale_scores.device)
#     sum_exp = F.conv2d(exp, weight=weight, padding=padding).clamp(min=1e-8)
#
#     return exp / sum_exp
#
#
# def multi_scale_softmax(multi_scale_scores, strength=100.0):
#     max_score, _ = multi_scale_scores.max(dim=1, keepdim=True)
#
#     exp = torch.exp(strength * (multi_scale_scores - max_score))
#     sum_exp = exp.sum(dim=1, keepdim=True).clamp(min=1e-8)
#     softmax = exp / sum_exp
#
#     score = torch.sum(multi_scale_scores * softmax, dim=1, keepdim=True)
#
#     return score
#
#
# class DepthToSpace(nn.Module):
#
#     def __init__(self, grid_size):
#         super().__init__()
#         self.grid_size = grid_size
#         self.grid_size_sq = grid_size * grid_size
#
#     def forward(self, x):
#         output = x.permute(0, 2, 3, 1)
#         (batch_size, d_height, d_width, d_depth) = output.size()
#
#         s_depth = int(d_depth / self.grid_size_sq)
#         s_width = int(d_width * self.grid_size)
#         s_height = int(d_height * self.grid_size)
#         t_1 = output.reshape(batch_size, d_height, d_width, self.grid_size_sq, s_depth)
#         spl = t_1.split(self.grid_size, 3)
#         stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
#
#         output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
#                                                                                       s_depth)
#         output = output.permute(0, 3, 1, 2)
#
#         return output
#
#
# class SpaceToDepth(nn.Module):
#
#     def __init__(self, grid_size):
#         super().__init__()
#         self.grid_size = grid_size
#
#     def forward(self, x):
#         """
#         :param x: B x C x H x W
#         :param grid_size: int
#         :return B x C * grid_size^2 x Hr x Wr :type torch.tensor, any
#         """
#         b, c, h, w = x.size()
#
#         hr = h // self.grid_size
#         wr = w // self.grid_size
#
#         x = x.view(b, c, hr, self.grid_size, wr, self.grid_size)
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # B x grid_size x grid_size x C x Hr x Wr
#
#         return x.view(b, c * (self.grid_size ** 2), hr, wr)
