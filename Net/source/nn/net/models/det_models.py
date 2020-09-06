import torch.nn as nn

import torch.nn.functional as F

import Net.source.core.experiment as exp

from Net.source.nn.net.utils.model_utils import vgg_block, spatial_soft_nms, normalize_conf_score


class DetJointBranch(nn.Module):

    @staticmethod
    def from_config(model_config):
        return DetJointBranch(model_config[exp.SOFT_NMS_KERNEL_SIZE], model_config[exp.BATCH_NORM])

    def __init__(self, soft_nms_kernel_size, batch_norm):
        super().__init__()
        self.soft_nms_kernel_size = soft_nms_kernel_size

        self.conf_conv1 = vgg_block(128, 256, batch_norm=batch_norm)
        self.conf_conv2 = nn.Conv2d(256, 1, kernel_size=1, padding=0)

        self.sal_conv1 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x, x4):
        x = self.conf_conv1(x)
        log_conf_score = self.conf_conv2(x)

        b, _, hc, wc = log_conf_score.shape

        conf_score = F.softmax(log_conf_score.view(b, -1), -1).view(b, 1, hc, wc)

        log_sal_score = self.sal_conv1(x4)
        sal_score = spatial_soft_nms(log_sal_score, self.soft_nms_kernel_size)

        score = sal_score * normalize_conf_score(conf_score)

        return score, conf_score, log_conf_score, sal_score


# # Legacy models
#
# class DetMSDBranch(nn.Module):
#
#     @staticmethod
#     def from_config(model_config):
#         return DetMSDBranch(model_config[exp.SOFT_NMS_KERNEL_SIZE],
#                             model_config[exp.GRID_SIZE])
#
#     def __init__(self, soft_nms_kernel_size, grid_size):
#         super().__init__()
#
#         self.soft_nms_kernel_size = soft_nms_kernel_size
#
#         self.det_ms = make_ms_detector(128, 256, grid_size)
#
#     def forward(self, x, log_ms_score):
#         s = self.det_ms(x)
#         log_ms_score = torch.cat(log_ms_score + [s], dim=1)
#
#         nms_ms_scores = multi_scale_nms(log_ms_score, self.soft_nms_kernel_size)
#         score = multi_scale_softmax(nms_ms_scores)
#
#         model_info = {
#             mu.LOG_MS_SCORE: log_ms_score,
#             mu.NMS_MS_SCORE: nms_ms_scores,
#         }
#
#         return score, model_info
#
#
# class DetMSBranch(nn.Module):
#
#     @staticmethod
#     def from_config(model_config):
#         return DetMSBranch(model_config[exp.SOFT_NMS_KERNEL_SIZE])
#
#     def __init__(self, soft_nms_kernel_size):
#         super().__init__()
#         self.soft_nms_kernel_size = soft_nms_kernel_size
#
#     def forward(self, x, log_ms_score):
#         log_ms_score = torch.cat(log_ms_score, dim=1)
#
#         nms_ms_scores = multi_scale_nms(log_ms_score, self.soft_nms_kernel_size)
#         score = multi_scale_softmax(nms_ms_scores)
#
#         model_info = {
#             mu.LOG_MS_SCORE: log_ms_score,
#             mu.NMS_MS_SCORE: nms_ms_scores,
#         }
#
#         return score, model_info
#
#
# class DetJointOldBranch(nn.Module):
#
#     def forward(self, sal_score, conf_score):
#         b = sal_score.shape[0]
#
#         # Need a more sophisticated method of normalization
#
#         conf_score = conf_score - conf_score.view(b, -1).min(dim=-1)[0].view(b, 1, 1, 1)
#         conf_score = conf_score / conf_score.view(b, -1).max(dim=-1)[0].view(b, 1, 1, 1)
#         conf_score = F.interpolate(conf_score, scale_factor=4.0).detach()
#
#         return sal_score * conf_score
#
#
# class DetSalOldBranch(nn.Module):
#
#     @staticmethod
#     def from_config(model_config):
#         return DetSalOldBranch(model_config[exp.SOFT_NMS_KERNEL_SIZE])
#
#     def __init__(self, soft_nms_kernel_size):
#         super().__init__()
#         self.det_sal = make_sal_detector_block(64, 1, True)
#
#         self.soft_nms_kernel_size = soft_nms_kernel_size
#
#     def forward(self, x):
#         log_sal_score = self.det_sal(x)
#
#         sal_score = spatial_soft_nms(log_sal_score, self.soft_nms_kernel_size)
#
#         return sal_score
#
#
# class DetConfOldBranch(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.det_conf = make_conf_detector_block(128, 256)
#
#     def forward(self, x):
#         log_conf_score = self.det_conf(x)
#
#         b, _, h, w = log_conf_score.shape
#
#         conf_score = F.softmax(log_conf_score.view(b, -1), -1).view(b, 1, h, w)
#
#         return conf_score, log_conf_score
