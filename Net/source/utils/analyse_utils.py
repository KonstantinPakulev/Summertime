# import torch
#
# from Net.source.nn.net.utils.endpoint_utils import sample_descriptors
# from Net.source.utils.matching_utils import calculate_descriptor_distance, get_mutual_matches, DescriptorDistance
#
#
# ANALYSIS_DATA = 'analysis_data'
#
# POS_DESC_DIST = 'pos_desc_dist'
# NEG_DESC_DIST = 'neg_desc_dist'
#
# POS_DESC_DIST2 = 'pos_desc_dist2'
# NEG_DESC_DIST2 = 'neg_desc_dist2'
#
# POS_NUM = 'pos_num'
# VIS_NUM = 'vis_num'
#
#
# def measure_pos_neg_desc_dist(desc1, desc2, w_desc_grid1, w_vis_desc_grid_mask1, grid_size):
#     b, c, _, _ = desc1.shape
#
#     _desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
#     w_desc1 = sample_descriptors(desc2, w_desc_grid1, grid_size)
#
#     desc_dist = calculate_descriptor_distance(_desc1, w_desc1, DescriptorDistance.INV_COS_SIM)
#
#     nn_desc_value1, nn_desc_ids1 = desc_dist.topk(dim=-1, k=2, largest=False)
#     nn_desc_ids2 = desc_dist.min(dim=-2)[1]
#
#     mdm_mask = get_mutual_matches(nn_desc_ids1[..., 0], nn_desc_ids2)
#     ids = torch.arange(0, mdm_mask.shape[1]).repeat(mdm_mask.shape[0], 1).to(mdm_mask.device)
#     correct_mask = nn_desc_ids1[..., 0] == ids
#
#     desc_mask1 = mdm_mask * correct_mask * w_vis_desc_grid_mask1
#     n_desc_mask1 = ~(mdm_mask * correct_mask) * w_vis_desc_grid_mask1
#
#     pos_dist = (nn_desc_value1[..., 0] * desc_mask1.float()).sum(dim=-1) / desc_mask1.float().sum(dim=-1).clamp(min=1e-8)
#     neg_dist = (nn_desc_value1[..., 0] * n_desc_mask1.float()).sum(dim=-1) / n_desc_mask1.float().sum(dim=-1).clamp(min=1e-8)
#
#     pos_second_dist = (nn_desc_value1[..., 1] * desc_mask1.float()).sum(dim=-1) / desc_mask1.float().sum(dim=-1).clamp(min=1e-8)
#     neg_second_dist = (nn_desc_value1[..., 1] * n_desc_mask1.float()).sum(dim=-1) / n_desc_mask1.float().sum(dim=-1).clamp(min=1e-8)
#
#     pos_num = (mdm_mask * correct_mask).sum(dim=-1)
#     vis_num = w_vis_desc_grid_mask1.sum(dim=-1)
#
#     return pos_dist, neg_dist, pos_second_dist, neg_second_dist, pos_num, vis_num
#
