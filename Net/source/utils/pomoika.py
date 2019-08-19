# def multi_scale_nms(multi_scale_scores, k_size, strength=3.0):
#     padding = k_size // 2
#
#     nms_scale_scores = F.max_pool2d(multi_scale_scores, kernel_size=k_size, padding=padding, stride=1)
#     max_scale_scores, _ = nms_scale_scores.max(dim=1)
#
#     _, c, _, _ = multi_scale_scores.size()
#
#     exp = torch.exp(strength * (multi_scale_scores - max_scale_scores.unsqueeze(1)))
#     weight = torch.ones((1, c, k_size, k_size)).to(multi_scale_scores.device)
#     sum_exp = F.conv2d(exp, weight=weight, padding=padding) + 1e-8
#
#     return exp / sum_exp
#
#
# def multi_scale_softmax(multi_scale_scores, strength=100.0):
#     max_scores, _ = multi_scale_scores.max(dim=1, keepdim=True)
#
#     exp = torch.exp(strength * (multi_scale_scores - max_scores))
#     sum_exp = exp.sum(dim=1, keepdim=True) + 1e-8
#     softmax = exp / sum_exp
#
#     score = torch.sum(multi_scale_scores * softmax, dim=1, keepdim=True)
#
#     return score