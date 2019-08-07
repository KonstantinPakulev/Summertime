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