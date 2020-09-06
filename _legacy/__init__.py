# class MSELoss_ver2(nn.Module):
#
#     @staticmethod
#     def from_config(model_config, criterion_config):
#         return MSELoss_ver2(model_config[NMS_THRESH], model_config[NMS_K_SIZE],
#                             model_config[TOP_K],
#                             criterion_config[GAUSS_K_SIZE], criterion_config[GAUSS_SIGMA],
#                             criterion_config[DET_LAMBDA])
#
#     def __init__(self, nms_thresh, nms_k_size, top_k, gauss_k_size, gauss_sigma, loss_lambda):
#         super().__init__()
#
#         self.nms_thresh = nms_thresh
#         self.nms_k_size = nms_k_size
#
#         self.top_k = top_k
#
#         self.gauss_k_size = gauss_k_size
#         self.gauss_sigma = gauss_sigma
#
#         self.loss_lambda = loss_lambda
#
#
#     def forward(self, score1, w_score2, w_vis_mask2):
#         """
#         :param score1: B x 1 x H x W
#         :param w_score2: B x 1 x H x W
#         :param w_vis_mask2: B x 1 x H x W
#         :return: float
#         """
#         b = score1.size(0)
#
#         gt_score1 = prepare_gt_score_ver2(w_score2, self.nms_thresh, self.nms_k_size, self.top_k)
#
#         loss = F.mse_loss(score1, gt_score1, reduction='none') * w_vis_mask2.float()  # B x 1 x H x W
#         loss = loss.view(b, -1).sum(dim=-1) / w_vis_mask2.float().view(b, -1).sum(dim=-1).clamp(min=1e-8)  # B
#         loss = self.loss_lambda * loss.mean()
#
#         return loss