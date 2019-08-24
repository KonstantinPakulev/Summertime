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


# b, n, _ = kp1.size()
#
# g_ids = ids.unsqueeze(dim=-1).repeat((1, 1, 2))
# w_kp2 = w_kp2.gather(dim=1, index=g_ids)
# kp2 = kp2.gather(dim=1, index=g_ids)
#
# f_kp1 = kp1.float().view(b * n, 2)
# f_kp2 = w_kp2.float().view(b * n, 2)
#
# kp1_mask = f_kp1.sum(dim=-1).ne(0.0)
# kp2_mask = f_kp2.sum(dim=-1).ne(0.0)
#
# dist = torch.pairwise_distance(f_kp1, f_kp2)
# correct_matches = dist.le(kp_match_thresh) * kp1_mask * kp2_mask
# correct_matches = correct_matches.view(b, n, 1)
#
# kp1 = kp1 * correct_matches.long()
# kp2 = kp2 * correct_matches.long()
#
# if mask1 is not None:
#     score = (correct_matches.squeeze(-1).float().sum(dim=-1) / mask1.sum(dim=1).float()).mean()
#     return score, kp1, kp2
# else:
#     return kp1, kp2
# def multi_scale_nms_softmax(multi_scale_scores, ks):
#     exp_score = torch.exp(multi_scale_scores)
#     weight = torch.ones((exp_score.size(1), exp_score.size(1), ks, ks)).to(multi_scale_scores.device)
#     exp_sum = F.conv2d(exp_score, weight=weight, padding=ks // 2) + 1e-8
#
#     score_nms = exp_score / exp_sum
#
#     exp_score = torch.exp(score_nms)
#     exp_sum = exp_score.sum(dim=1, keepdim=True) + 1e-8
#
#     ms_score = exp_score / exp_sum
#
#     ms_score = torch.sum(multi_scale_scores * ms_score, dim=1, keepdim=True)
#
#     return ms_score

# if unique:
#     # Search for closest unique matches
#     unique_matches = torch.ones((b, n2)).to(w_kp2.device) * kp_match_thresh
#     if provide_kp:
#         backward_mapping = torch.ones((b, n2), dtype=torch.long) * -1
#     for i in b_ids:
#         for j in kp1_ids:
#             if nn_values[i, j].lt(unique_matches[i, nn_ids[i, j]]) and (mask1 is None or mask1[i, j]):
#                 unique_matches[i, nn_ids[i, j]] = nn_values[i, j]
#                 if provide_kp:
#                     backward_mapping[i, nn_ids[i, j]] = j
#     matches = unique_matches.lt(kp_match_thresh).float()
# else:
#     # Don't consider uniqueness
#     matches = torch.zeros((b, n2)).to(w_kp2.device)
#     for i in b_ids:
#         for j in kp1_ids:
#             if mask1 is None or mask1[i, j]:
#                 matches[i, nn_ids[i, j]] += nn_values[i, j].lt(kp_match_thresh).float()
#
# match_mask2 = matches * visible1.float()
# correct_matches = match_mask2.sum(dim=1)
# num_visible = visible1.sum(dim=1)
#
# if provide_kp:
#     kp2_ids = torch.arange(n2)
#     match_mask1 = torch.zeros((b, n1), dtype=torch.uint8).to(kp1.device)
#     for i in b_ids:
#         for j in kp2_ids:
#             if backward_mapping[i, j].ne(-1) and visible1[i, j]:
#                 match_mask1[i, backward_mapping[i, j]] = 1
#
# rep_score = (correct_matches.float() / num_visible.float()).mean()
#
# if provide_kp:
#     assert b == 1 and unique
#     m_kp1 = kp1 * match_mask1.unsqueeze(-1).long()
#     m_kp2 = kp2 * match_mask2.unsqueeze(-1).long()
#     m_ids = nn_ids * match_mask1.long()
#     return rep_score, m_kp1, m_kp2, m_ids
# else:
#     return rep_score
# class HardTripletLoss(nn.Module):
#

#
#     def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
#         """
#         :param kp1 B x N x 2
#         :param w_kp1: B x N x 2
#         :param kp1_desc: B x N x C
#         :param desc2: B x C x H x W
#         :param homo12: B x 3 x 3
#         :return: float
#         """
#         b, n, c = kp1_desc.size()
#
#         w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)
#
#         # Take positive matches
#         positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
#         positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, self.num_neg).view(b, n * self.num_neg)
#
#         # Create neighbour mask
#         coo_grid = create_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
#
#         d1_inter_num = 4
#
#         grid_dist1 = calculate_distance_matrix(kp1, coo_grid)
#         _, kp1_cell_ids = grid_dist1.topk(k=d1_inter_num, largest=False, dim=-1)
#
#         kp1_cell_ids = kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
#         kp1_cells = coo_grid.gather(dim=1, index=kp1_cell_ids)
#
#         w_kp1_cells = warp_points(kp1_cells, homo12)
#
#         d2_inter_num = 4
#
#         grid_dist2 = calculate_distance_matrix(w_kp1_cells, coo_grid)
#         _, kp2_cell_ids = grid_dist2.topk(k=d2_inter_num, largest=False, dim=-1)
#
#         neigh_mask = torch.zeros_like(grid_dist2).to(grid_dist2)
#         neigh_mask = neigh_mask.scatter(dim=-1, index=kp2_cell_ids, value=1)
#         neigh_mask = neigh_mask.view(b, n, d2_inter_num, -1).sum(dim=2).float()
#
#         # Calculate similarity
#         desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
#         desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)
#
#         #  Apply neighbour mask and get negatives
#         desc_sim = desc_sim + neigh_mask * 5
#         neg_sim = desc_sim.topk(k=self.num_neg, dim=-1, largest=False)[0].view(b, -1)
#
#         loss = torch.clamp(positive_sim - neg_sim + self.margin, min=0).mean() * self.loss_lambda
#
#         return loss
#
#
# class HardQuadTripletLoss(HardTripletLoss):
#
#     def __init__(self, grid_size, margin, num_neg, loss_lambda):
#         super().__init__(grid_size, margin, num_neg, loss_lambda)
#
#     def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
#         """
#         :param kp1 B x N x 2
#         :param w_kp1: B x N x 2
#         :param kp1_desc: B x N x C
#         :param desc2: B x C x H x W
#         :param homo12: B x 3 x 3
#         :return: float
#         """
#         b, n, c = kp1_desc.size()
#
#         w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)
#
#         # Take positive matches
#         positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
#         positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, self.num_neg).view(b, n * self.num_neg)
#
#         # Create neighbour mask
#         coo_grid = create_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
#
#         d1_inter_num = 4
#
#         grid_dist1 = calculate_distance_matrix(kp1, coo_grid)
#         _, kp1_cell_ids = grid_dist1.topk(k=d1_inter_num, largest=False, dim=-1)
#
#         kp1_cell_ids = kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
#         kp1_cells = coo_grid.gather(dim=1, index=kp1_cell_ids)
#
#         w_kp1_cells = warp_points(kp1_cells, homo12)
#
#         d2_inter_num = 4
#
#         grid_dist2 = calculate_distance_matrix(w_kp1_cells, coo_grid)
#         _, kp2_cell_ids = grid_dist2.topk(k=d2_inter_num, largest=False, dim=-1)
#
#         neigh_mask = torch.zeros_like(grid_dist2).to(grid_dist2)
#         neigh_mask = neigh_mask.scatter(dim=-1, index=kp2_cell_ids, value=1)
#         neigh_mask = neigh_mask.view(b, n, d2_inter_num, -1).sum(dim=2).float()
#
#         # Calculate similarity
#         desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
#         desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)
#
#         #  Apply neighbour mask and get negatives
#         desc_sim = desc_sim + neigh_mask * 5
#         neg_sim = desc_sim.topk(k=self.num_neg, dim=-1, largest=False)[0].view(b, -1)
#
#         loss = (torch.clamp(positive_sim - neg_sim + self.margin, min=0)**2).mean() * self.loss_lambda
#
#         return loss