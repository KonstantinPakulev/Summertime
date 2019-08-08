# dot_desc -= s * 3
# dot_desc = dot_desc.view(n, -1, hr * wr)
#
# # Calculate loss
# dot_pos = (kp1_desc * pos_desc).sum(dim=-1)
# dot_neg = dot_desc.max(dim=-1)[0]
#
# loss = torch.clamp(dot_pos - dot_neg + self.margin, min=0.0).mean() * self.loss_lambda

# class HomoHingeLoss(nn.Module):
#
#     def __init__(self, grid_size, pos_lambda, pos_margin, neg_margin, loss_lambda):
#         super().__init__()
#         self.grid_size = grid_size
#
#         self.pos_lambda = pos_lambda
#
#         self.pos_margin = pos_margin
#         self.neg_margin = neg_margin
#
#         self.loss_lambda = loss_lambda
#
#     def forward(self, desc1, desc2, homo21, vis_mask1):
#         """
#         :param desc1: N x C x Hr x Wr
#         :param desc2: N x C x Hr x Wr
#         :param homo21: N x 3 x 3
#         :param vis_mask1: Mask of the first image. N x 1 x H x W
#         Note: 'r' suffix means reduced in 'grid_size' times
#         """
#         # Because desc is in reduced coordinate space we need to create a mapping to original coordinates
#         grid = create_coordinates_grid(desc1.size()) * self.grid_size + self.grid_size // 2
#         grid = grid.type_as(desc1).to(desc1.device)
#         w_grid = warp_coordinates_grid(grid, homo21)
#
#         grid = grid.unsqueeze(dim=3).unsqueeze(dim=3)
#         w_grid = w_grid.unsqueeze(dim=1).unsqueeze(dim=1)
#
#         n, _, hr, wr = desc1.size()
#
#         # Reduce spatial dimensions of visibility mask
#         vis_mask1 = space_to_depth(vis_mask1, self.grid_size).prod(dim=1)
#         vis_mask1 = vis_mask1.reshape([n, 1, 1, hr, wr])
#
#         # Mask with homography induced correspondences
#         grid_dist = torch.norm(grid - w_grid, dim=-1)
#         ones = torch.ones_like(grid_dist)
#         zeros = torch.zeros_like(grid_dist)
#         s = torch.where(grid_dist <= self.grid_size - 0.5, ones, zeros)
#
#         ns = 1 - s
#
#         # Apply visibility mask
#         s *= vis_mask1
#         ns *= vis_mask1
#
#         desc1 = desc1.unsqueeze(4).unsqueeze(4)
#         desc2 = desc2.unsqueeze(2).unsqueeze(2)
#         dot_desc = torch.sum(desc1 * desc2, dim=1)
#
#         pos_dist = (self.pos_margin - dot_desc).clamp(min=0)
#         neg_dist = (dot_desc - self.neg_margin).clamp(min=0)
#
#         loss = self.pos_lambda * s * pos_dist + ns * neg_dist
#
#         norm = hr * wr * vis_mask1.sum()
#         loss = loss.sum() / norm * self.loss_lambda
#
#         return loss
# class HardHingeLoss(nn.Module):
#
#     def __init__(self, grid_size,
#                  pos_margin, neg_margin,
#                  loss_lambda):
#         super().__init__()
#         self.grid_size = grid_size
#
#         self.pos_margin = pos_margin
#         self.neg_margin = neg_margin
#
#         self.loss_lambda = loss_lambda
#
#     def forward(self, kp1, desc1, desc2, homo21, vis_mask1):
#         """
#         :param kp1: K x 4
#         :param desc1: N x C x Hr x Wr
#         :param desc2: N x C x Hr x Wr
#         :param homo21: N x 3 x 3
#         :param vis_mask1: Mask of the first image. N x 1 x H x W
#         Note: 'r' suffix means reduced in 'grid_size' times
#         """
#
#         # Move keypoints coordinates to reduced spatial size
#         kp_grid = kp1[:, [3, 2]].unsqueeze(0).float()
#         kp_grid[:, 0] = kp_grid[:, 0] / self.grid_size
#         kp_grid[:, 1] = kp_grid[:, 1] / self.grid_size
#
#         # Warp reduced coordinate grid to desc1 viewpoint
#         w_grid = create_coordinates_grid(desc2.size()) * self.grid_size + self.grid_size // 2
#         w_grid = w_grid.type_as(desc2).to(desc2.device)
#         w_grid = warp_coordinates_grid(w_grid, homo21)
#
#         kp_grid = kp_grid.unsqueeze(2).unsqueeze(2)
#         w_grid = w_grid.unsqueeze(1)
#
#         n, _, hr, wr = desc1.size()
#
#         # Reduce spatial dimensions of visibility mask
#         vis_mask1 = space_to_depth(vis_mask1, self.grid_size).prod(dim=1)
#         vis_mask1 = vis_mask1.reshape([n, 1, hr, wr])
#
#         # Mask with homography induced correspondences
#         grid_dist = torch.norm(kp_grid - w_grid, dim=-1)
#         s = (grid_dist <= self.grid_size - 0.5).float()
#
#         ns = 1 - s
#
#         # Apply visibility mask
#         s *= vis_mask1
#         ns *= vis_mask1
#
#         # Sample descriptors
#         kp1_desc = sample_descriptors(desc1, kp1, self.grid_size)
#
#         # Calculate distance pairs
#         s_kp1_desc = kp1_desc.permute(1, 0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
#         s_desc2 = desc2.unsqueeze(2)
#         dot_desc = torch.sum(s_kp1_desc * s_desc2, dim=1)
#
#         pos_dist = (self.pos_margin - dot_desc).clamp(min=0)
#         neg_dist = (dot_desc - self.neg_margin).clamp(min=0)
# Move keypoints coordinates to reduced spatial size
# # TODO. SHTO ETO ZA HUINYA, BRATIK?
# kp_grid = kp1[:, [3, 2]].unsqueeze(0).float()
# kp_grid[:, 0] = kp_grid[:, 0] / self.grid_size
# kp_grid[:, 1] = kp_grid[:, 1] / self.grid_size
#
# # Warp reduced coordinate grid to desc1 viewpoint
# w_grid = create_coordinates_grid(desc2.size()) * self.grid_size + self.grid_size // 2
# w_grid = w_grid.type_as(desc2).to(desc2.device)
# w_grid = warp_coordinates_grid(w_grid, homo21)
#
# kp_grid = kp_grid.unsqueeze(2).unsqueeze(2)
# w_grid = w_grid.unsqueeze(1)
#
# n, _, hr, wr = desc1.size()
#
# # Reduce spatial dimensions of visibility mask
# vis_mask1 = space_to_depth(vis_mask1, self.grid_size).prod(dim=1)
# vis_mask1 = vis_mask1.reshape([n, 1, hr, wr])
#
# # Mask with homography induced correspondences
# grid_dist = torch.norm(kp_grid - w_grid, dim=-1)
# ones = torch.ones_like(grid_dist)
# zeros = torch.zeros_like(grid_dist)
# s = torch.where(grid_dist <= self.grid_size - 0.5, ones, zeros)
#
# ks1, ks2, ks3 = 31, 11, 5
# sigma1, sigma2, sigma3 = 7, 4, 2
#
# ns = s.clone().view(-1, 1, hr, wr)
# ns1 = gaussian_filter(ns, ks1, sigma1).view(n, kp1.size(0), hr, wr) - s
# ns2 = gaussian_filter(ns, ks2, sigma2).view(n, kp1.size(0), hr, wr) - s
# ns3 = gaussian_filter(ns, ks3, sigma3).view(n, kp1.size(0), hr, wr) - s
# ns = ns1 + ns2 + ns3
#
# # Apply visibility mask
# s *= vis_mask1
# ns *= vis_mask1
#
# # Sample descriptors
# kp1_desc = sample_descriptors(desc1, kp1, self.grid_size)
#
# # Calculate distance pairs
# s_kp1_desc = kp1_desc.permute(1, 0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
# s_desc2 = desc2.unsqueeze(2)
# dot_desc = torch.sum(s_kp1_desc * s_desc2, dim=1)
#
# pos_dist = (self.pos_margin - dot_desc).clamp(min=0)
# neg_dist = (dot_desc - self.neg_margin).clamp(min=0)
#
# neg_per_pos = 374.359
# pos_lambda = neg_per_pos * 0.3
#
# loss = pos_lambda * s * pos_dist + ns * neg_dist
#
# norm = kp1.size(0) * neg_per_pos
# loss = loss.sum() / norm * self.loss_lambda
#
# return loss, kp1_desc

# class HingeLoss(nn.Module):
#
#     def __init__(self, grid_size,
#                  pos_margin, neg_margin,
#                  neg_samples,
#                  loss_lambda):
#         super().__init__()
#         self.grid_size = grid_size
#
#         self.pos_margin = pos_margin
#         self.neg_margin = neg_margin
#
#         self.neg_samples = neg_samples
#
#         self.loss_lambda = loss_lambda
#
#     def forward(self, kp1, w_kp1, kp2, kp1_desc, kp2_desc, desc2):
#         w_kp1_grid = w_kp1.float().unsqueeze(1)
#         kp2_grid = kp2.float().unsqueeze(0)
#
#         grid_dist = torch.norm(w_kp1_grid - kp2_grid, dim=-1)
#         radius = math.sqrt(self.grid_size ** 2 / 2)
#         neighbour_mask = (grid_dist <= 2 * radius + 0.1).float()
#
#         positive = sample_descriptors(desc2, w_kp1, self.grid_size)
#
#         # Cosine similarity measure
#         desc_dot = torch.mm(kp1_desc, kp2_desc.t())
#         desc_dot = desc_dot - neighbour_mask * 5
#
#         positive_dot = (kp1_desc * positive).sum(dim=-1)
#         negative_dot = desc_dot.topk(self.neg_samples, dim=-1)[0]
#
#         balance_factor = self.neg_samples
#         loss = torch.clamp(self.pos_margin - positive_dot, min=0).sum() * balance_factor + \
#                torch.clamp(negative_dot - self.neg_margin, min=0).sum()
#
#         norm = self.loss_lambda / (kp1_desc.size(0) * self.neg_samples)
#         loss = loss * norm
#
#         return loss

# class HardTripletLoss(nn.Module):
#
#     def __init__(self, grid_size, margin, loss_lambda):
#         super().__init__()
#
#         self.grid_size = grid_size
#
#         self.margin = margin
#
#         self.loss_lambda = loss_lambda
#
#     def forward(self, kp1, w_kp1, kp2, kp1_desc, kp2_desc, desc2):
#         w_kp1_grid = w_kp1.float().unsqueeze(1)
#         kp2_grid = kp2.float().unsqueeze(0)
#
#         grid_dist = torch.norm(w_kp1_grid - kp2_grid, dim=-1)
#         radius = math.sqrt(self.grid_size ** 2 / 2)
#         neighbour_mask = (grid_dist <= 2 * radius + 0.1).float()
#
#         anchor = kp1_desc.unsqueeze(1)
#         positive = sample_descriptors(desc2, w_kp1, self.grid_size)
#         negative = kp2_desc.unsqueeze(0)
#
#         # L2 distance measure
#         desc_dist = torch.norm(anchor - negative, dim=-1)
#         desc_dist = desc_dist + neighbour_mask * 5
#
#         positive_dist = torch.pairwise_distance(kp1_desc, positive)
#         # Pick closest negative sample
#         negative_dist = desc_dist.min(dim=-1)[0]
#
#         loss = torch.clamp(positive_dist - negative_dist + self.margin, min=0).mean() * self.loss_lambda
#
#         return loss


# def forward(self, kp1, w_kp1, kp1_desc, desc2):
#     # radius = math.sqrt(self.grid_size ** 2 / 2)
#     radius = self.grid_size / 2
#
#     # Anchor mask
#     anchor_grid1 = kp1.float().unsqueeze(1)
#     anchor_grid2 = kp1.float().unsqueeze(0)
#     anchor_dist = torch.norm(anchor_grid1 - anchor_grid2, dim=-1)
#     anchor_mask = (anchor_dist <= 2 * radius + 0.1).float()
#
#     # Positive mask
#     pos_grid1 = w_kp1.float().unsqueeze(1)
#     pos_grid2 = w_kp1.float().unsqueeze(0)
#     pos_dist = torch.norm(pos_grid1 - pos_grid2, dim=-1)
#     pos_mask = (pos_dist <= 2 * radius + 0.1).float()
#
#     # Diagonal mask
#     diag_mask = torch.eye(kp1.size(0)).to(kp1.device)
#
#     # Calculate similarity measure between anchors and positives
#     w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)
#     desc_sim = calculate_similarity_measure(kp1_desc, w_kp1_desc)
#
#     # Take positive matches
#     positive_sim = desc_sim.diag()
#
#     # Find closest negatives, but with accounting for neighbours
#     desc_sim = desc_sim + diag_mask * 5
#     desc_sim = desc_sim + anchor_mask * 5
#     desc_sim = desc_sim + pos_mask * 5
#
#     anchor_neg_sim = desc_sim.min(dim=-1)[0]
#     pos_neg_sim = desc_sim.min(dim=0)[0]
#     neg_sim = torch.min(anchor_neg_sim, pos_neg_sim)
#
#     loss = torch.clamp(positive_sim - neg_sim + self.margin, min=0).mean() * self.loss_lambda
#
#     return loss

# row_idx = torch.arange(0, kp1.size(0)).view(kp1.size(0), 1).repeat([1, 4]).view(-1)
#
#         # Anchor mask
#         anchor_grid1 = kp1.float().unsqueeze(1)
#         anchor_grid2 = kp1.float().unsqueeze(0)
#         anchor_dist = torch.norm(anchor_grid1 - anchor_grid2, dim=-1)
#         a_idx = anchor_dist.topk(k=4, dim=-1, largest=False)[1].view(-1)
#
#         anchor_mask = torch.zeros_like(anchor_dist)
#         anchor_mask[row_idx, a_idx] = 1.0
#         anchor_mask[a_idx, row_idx] = 1.0
#
#         # Positive mask
#         pos_grid1 = w_kp1.float().unsqueeze(1)
#         pos_grid2 = w_kp1.float().unsqueeze(0)
#         pos_dist = torch.norm(pos_grid1 - pos_grid2, dim=-1)
#         p_idx = pos_dist.topk(k=4, dim=-1, largest=False)[1].view(-1)
#
#         pos_mask = torch.zeros_like(pos_dist)
#         pos_mask[row_idx, p_idx] = 1.0
#         pos_mask[p_idx, row_idx] = 1.0
#
#         # Diagonal mask
#         diag_mask = torch.eye(kp1.size(0)).to(kp1.device)
#
#         # Calculate similarity measure between anchors and positives
#         w_kp1_desc = sample_descriptors(desc2, w_kp1, self.grid_size)
#         desc_sim = calculate_similarity_measure(kp1_desc, w_kp1_desc)
#
#         # Take positive matches
#         positive_sim = desc_sim.diag()
#
#         # Find closest negatives, but with accounting for neighbours
#         desc_sim = desc_sim + diag_mask * 5
#         desc_sim = desc_sim + anchor_mask * 5
#         desc_sim = desc_sim + pos_mask * 5
#
#         anchor_neg_sim = desc_sim.min(dim=-1)[0]
#         pos_neg_sim = desc_sim.min(dim=0)[0]
#         neg_sim = torch.min(anchor_neg_sim, pos_neg_sim)
#
#         loss = torch.clamp(positive_sim - neg_sim + self.margin, min=0).mean() * self.loss_lambda
#
#         return loss

# kp_grid = kp2coord(kp1).unsqueeze(1)
# grid = create_coordinates_grid(desc2.size()).view(-1, 2) * grid_size + grid_size // 2
# grid_dist = torch.norm(kp_grid - grid, dim=-1)
# nn_cells = grid_dist.topk(k=9, largest=False, dim=-1)[1]
#
# #%%
# cv_image = torch2cv(image2)
#
# print(nn_cells)
# print(flat2grid(nn_cells, 30, 40)[900:909, :] * 8)
#
# cv_nn_cells = to_cv2_keypoint(flat2grid(nn_cells, 240, 320)[900:909, :])
# cv_selected_kp = to_cv2_keypoint(w_kp1[100].unsqueeze(0))
#
# cv_image = cv2.drawKeypoints(cv_image,
#                              cv_nn_cells, None, color=(0, 255, 0))
# cv_image = cv2.drawKeypoints(cv_image,
#                              cv_selected_kp, None, color=(255, 0, 0))
#
# #%%
#
# plot_figures({'cv_image2_kp': cv_image}, 1, 2)
# plt.imsave('neg_sampling_pictures/cv_image2_kp.png', cv_image)

#%%

