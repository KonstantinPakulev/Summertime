# import torch
#
#
# def distance_matrix(t1, t2):
#     """
#     :param t1: B x C x N
#     :param t2: B x C x N
#     :return d_matrix: B x N x N
#     """
#
#     d_matrix = 2 - 2 * torch.bmm(t1.transpose(1, 2), t2)  # [0, 4]
#     d_matrix = d_matrix.clamp(min=1e-8, max=4.0)
#     d_matrix = torch.sqrt(d_matrix)  # [0, 2]
#
#     return d_matrix


# print("Pred min distance index:", pmid)
# print("True min distance index:", tmid)
# print("Pred min distance:", dot_dest[0, index, pmid])
# print("True min distance:", dot_dest[0, index, tmid])
# print("All other distances in row", v)
# print("Their indicies:", ind[:20])
# print("In a short:", s[0, index, pmid])

# print("\n")
# print("EVAL")
#
# count = 0
#
# t1 = []
# t2 = []
# t3 = []
#
# for index in range(0, 1200):
#     pmid = dot_dest[0, index].argmax()
#     tmid = s[0, index].nonzero()
#
#     v, ind = torch.sort(dot_dest[0, index], descending=True)
#
#     if tmid in ind[:9]:
#         t1.append(index)
#         t2.append(pmid)
#
#         t3.append((pmid, tmid))
#         count += 1
#
# print("In a short:", count)


# dot = torch.zeros((1, 1, 240, 320))
# dot[0, 0, 120, 160] = 1
# d_dot = dilate_mask(dot)
# neg = d_dot - dot

# ones = torch.ones((1, 1, 240, 320))
# warped = warp_image(ones, homo).gt(0).float()
# w_eroded = erode_mask(warped)

# all = torch.cat((neg, warped, w_eroded, warped - w_eroded), dim=0)

# print(neg.nonzero().shape)

# writer.add_image("image", make_grid(all))

# mask1 = create_bordering_mask(im2, homo)
# mask2 = create_bordering_mask(im1, homo_inv)
#
# _, des1 = model(im1)
# _, des2 = model(im2)
#
# loss1, s1, dot_des1, r_mask1 = criterion(des1, des2, homo, mask1)
# loss2, s2, dot_des2, r_mask2 = criterion(des2, des1, homo_inv, mask2)
#
# loss = (loss1 + loss2) / 2
# loss.backward()


# return {
#     'loss': loss,
#
#     'des1': des1,
#     's1': s1,
#     'dot_des1': dot_des1,
#     'r_mask1': r_mask1,
#
#     'des2': des2,
#     's2': s2,
#     'dot_des2': dot_des2,
#     'r_mask2': r_mask2
# }
# \tValidation loss is {tester.state.metrics['loss']: .4f}
#                 \tNN match score is: {tester.state.metrics['nn_match']: .4f}

# image1, image2, homo = (
#     batch[IMAGE1].to(device),
#     batch[IMAGE2].to(device),
#     batch[HOMO].to(device)
# )

# score1, desc1 = model(image1)
# score2, desc2 = model(image2)
#
# score1 = score1.permute((0, 2, 3, 1))
# score2 = score2.permute((0, 2, 3, 1))
#
# score1 = filter_border(score1)
# score2 = filter_border(score2)
#
# # im1visible_mask = warp(torch.ones(score2.size()).to(homo.device), homo)
# # im2visible_mask = warp(torch.ones(score1.size()).to(homo.device), homo.inverse())
#
# kp1 = det_criterion.process(warp(score2, homo))[1].permute(0, 3, 1, 2).nonzero()
# kp2 = det_criterion.process(warp(score1, homo.inverse()))[1].permute(0, 3, 1, 2).nonzero()
#
# kp1_norm = det_criterion.gtscore(score2, homo)[1].permute(0, 3, 1, 2).nonzero()
#
# # desc1 = sample_descriptors(endpoint[DESC1], endpoint[KP1], cfg.MODEL.GRID_SIZE)
# # desc2 = sample_descriptors(endpoint[DESC2], endpoint[KP2], cfg.MODEL.GRID_SIZE)
#
# print(kp1)
# print(kp1_norm)
#
# # Score analysis
# w_score2 = warp(score2, homo)
#
# print(score1.unique())
# print(w_score2.unique())

# class HomoRFMSELoss(nn.Module):
#
#     def __init__(self, nms_thresh, nms_k_size, top_k, gauss_k_size, gauss_sigma):
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
#     def process(self, im1w_score):
#         """
#         nms(n), topk(t), gaussian kernel(g) operation
#         :param im1w_score: warped score map
#         :return: processed score map, topk mask, topk value
#         """
#
#         im1w_score = filter_border(im1w_score)
#
#         # apply nms to im1w_score
#         nms_mask = nms(im1w_score, self.nms_thresh, self.nms_k_size)
#         im1w_score = im1w_score * nms_mask
#         topk_value = im1w_score
#
#         # apply topk to im1w_score
#         topk_mask = topk_map(im1w_score, self.top_k)
#         im1w_score = topk_mask.to(torch.float) * im1w_score
#
#         # apply gaussian kernel to im1w_score
#         psf = im1w_score.new_tensor(
#             get_gauss_filter_weight(self.gauss_k_size, self.gauss_sigma)[
#             None, None, :, :
#             ]
#         )
#         im1w_score = F.conv2d(
#             input=im1w_score.permute(0, 3, 1, 2),
#             weight=psf,
#             stride=1,
#             padding=self.gauss_k_size // 2,
#         ).permute(
#             0, 2, 3, 1
#         )  # (B, H, W, 1)
#
#         """
#         apply tf.clamp to make sure all value in im1w_score isn't greater than 1
#         but this won't happend in correct way
#         """
#         im1w_score = im1w_score.clamp(min=0.0, max=1.0)
#
#         return im1w_score, topk_mask, topk_value
#
#     def gtscore(self, right_score, homolr):
#         im2_score = right_score
#         im2_score = filter_border(im2_score)
#
#         # warp im2_score to im1w_score and calculate visible_mask
#         im1w_score = warp(im2_score, homolr)
#         im1visible_mask = warp(
#             im2_score.new_full(im2_score.size(), fill_value=1, requires_grad=True),
#             homolr,
#         )
#
#         im1gt_score, topk_mask, topk_value = self.process(im1w_score)
#
#         return im1gt_score, topk_mask, topk_value, im1visible_mask
#
#     def loss(self, left_score, im1gt_score, im1visible_mask):
#         im1_score = left_score
#
#         l2_element_diff = (im1_score - im1gt_score) ** 2
#         # visualization numbers
#         Nvi = torch.clamp(im1visible_mask.sum(dim=(3, 2, 1)), min=2.0)
#         loss = (
#                 torch.sum(l2_element_diff * im1visible_mask, dim=(3, 2, 1)) / (Nvi + 1e-8)
#         ).mean()
#
#         return loss
#
#     def forward(self, score1, score2, homo):
#         score1 = score1.permute((0, 2, 3, 1))
#         score2 = score2.permute((0, 2, 3, 1))
#
#         im1_gtsc, im1_topkmask, im1_topkvalue, im1_visiblemask = self.gtscore(score2, homo)
#         im1_score, _, _  = self.process(score1)
#
#         im1_scloss = self.loss(im1_score, im1_gtsc, im1_visiblemask)
#
#
#         # im1mask_true = im1mask_true.permute(0, 3, 1, 2).nonzero()
#
#         # im_score = self.process(score1)[0]
#         # im_topk = topk_map(im_score, self.top_k)
#         kpts = im1_topkmask.permute(0, 3, 1, 2).nonzero()
#
#         return im1_scloss, kpts, im1_visiblemask

# v_adam,1.ppm,6.ppm,0.22223,-0.016421,140.29,-0.12556,0.6155,64.723,-0.00070518,-2.3579e-06,1.0016
# v_yuri,1.ppm,2.ppm,1.0035,-0.00055314,2.5255,-0.0028717,1.0087,-9.7285,-3.8783e-06,3.4244e-06,1
# v_woman,1.ppm,5.ppm,1.4351,-0.038,290.23,0.48702,1.2011,-120.76,0.0010757,2.4371e-05,0.99968
# class Net(nn.Module):
#
#     def __init__(self, grid_size, descriptor_size):
#         super().__init__()
#         self.grid_size = grid_size
#
#         self.backbone = make_vgg_backbone()
#         self.detector = make_vgg_detector_head(grid_size)
#         self.descriptor = make_vgg_descriptor_head(descriptor_size)
#
#     def forward(self, x):
#         """
#         :param x: B x C x H x W
#         """
#         x = self.backbone(x)
#         raw_score = self.detector(x)
#         raw_desc = self.descriptor(x)
#
#         raw_score = raw_score.softmax(dim=1)
#         # Remove 'no interest point' channel
#         raw_score = raw_score[:, :-1, :, :]
#         score = F.pixel_shuffle(raw_score, self.grid_size)
#
#         desc = F.normalize(raw_desc)
#
#         return score, desc
# def l_nn_match(x):
#     ms1 = nearest_neighbor_match_score(x['s1'], x['dot_des1'], x['r_mask1'])
#     ms2 = nearest_neighbor_match_score(x['s2'], x['dot_des2'], x['r_mask2'])
#     return (ms1 + ms2) / 2


# def l_nn_match_2(x):
#     ms1 = nearest_neighbor_match_score(x['s1'], x['dot_des1'], x['r_mask1'], 2)
#     ms2 = nearest_neighbor_match_score(x['s2'], x['dot_des2'], x['r_mask2'], 2)
#     return (ms1 + ms2) / 2


# def l_nn_match_4(x):
#     ms1 = nearest_neighbor_match_score(x['s1'], x['dot_des1'], x['r_mask1'], 4)
#     ms2 = nearest_neighbor_match_score(x['s2'], x['dot_des2'], x['r_mask2'], 4)
#     return (ms1 + ms2) / 2


# def l_nnt_match(x):
#     ms1 = nearest_neighbor_thresh_match_score(x['des1'], x['des2'], cfg.METRIC.THRESH,
#                                               x['s1'], x['dot_des1'], x['r_mask1'])
#     ms2 = nearest_neighbor_thresh_match_score(x['des2'], x['des1'], cfg.METRIC.THRESH,
#                                               x['s2'], x['dot_des2'], x['r_mask2'])
#     return (ms1 + ms2) / 2

# def l_nnr_match(x):
#     ms1 = nearest_neighbor_ratio_match_score(x['des1'], x['des2'], cfg.METRIC.RATIO,
#                                              x['s1'], x['dot_des1'], x['r_mask1'])
#     ms2 = nearest_neighbor_ratio_match_score(x['des2'], x['des1'], cfg.METRIC.RATIO,
#                                              x['s2'], x['dot_des2'], x['r_mask2'])
#     return (ms1 + ms2) / 2
# def collect_ids(dot_des, top_k):
#     """
#     :param dot_des: N x H x W x H x W. Matrix of distances
#     :param top_k: Consider if correct match is in k closest values
#     """
#     n, h, w, _, _ = dot_des.size()
#     flat = h * w
#
#     dot_des = dot_des.view(n, flat, flat)
#
#     k_flat = top_k * flat
#
#     ids0 = torch.ones((n, k_flat), dtype=torch.long) * torch.arange(0, n).view(n, 1)
#     ids1 = torch.ones((flat, top_k), dtype=torch.long) * torch.arange(0, flat).view(flat, 1)
#     ids1 = ids1.view(k_flat, 1).repeat((n, 1))
#     # A pair of normalized descriptors will have the maximum value of scalar product if they are the closets
#     _, ids2 = dot_des.topk(k=top_k, dim=-1)
#
#     n_flat = n * k_flat
#
#     ids0 = ids0.view(n_flat, 1).to(dot_des.device)
#     ids1 = ids1.view(n_flat, 1).to(dot_des.device)
#     ids2 = ids2.view(n_flat, 1)
#     ids = torch.cat((ids0, ids1, ids2), dim=-1)
#
#     return ids
#
#
# def get_correct_matches_indices(ids, s, r_mask):
#     """
#     :param ids: S x 3
#     :param s: N x H x W x H x W. Matrix of correspondences
#     :param r_mask: N x 1 x H x W
#     """
#     n, h, w, _, _ = s.size()
#     flat = h * w
#
#     s = s.view(n, flat, flat)
#
#     r_mask = r_mask.view(n, 1, flat)
#     s = s * r_mask
#
#     matches = s[ids[:, 0], ids[:, 1], ids[:, 2]]
#     cm_indices = matches.nonzero().squeeze(-1)
#
#     return cm_indices
#
#
# def get_unique_matches(cm_ids, r_mask):
#     """
#     :param cm_ids: S x 3
#     :param r_mask: N x 1 x H x W
#     """
#     n, _, h, w = r_mask.size()
#     flat = h * w
#
#     # Consider only unique matches between descriptors from the total number of possible matches
#     correct_matches = torch.zeros((flat,)).to(r_mask.device)
#     correct_matches[cm_ids[:, 2]] = 1
#
#     correct_matches = correct_matches.sum()
#     total = r_mask.sum()
#
#     return correct_matches / total
#
#
# def nearest_neighbor_match_score(s, dot_des, r_mask, top_k=1):
#     """
#     :param s: N x H x W x H x W. Matrix of correspondences
#     :param dot_des: N x H x W x H x W. Matrix of distances
#     :param r_mask: N x 1 x H x W
#     :param top_k: Consider if correct match is in k closest values
#     """
#
#     ids = collect_ids(dot_des, top_k)
#     cm_indices = get_correct_matches_indices(ids, s, r_mask)
#     cm_ids = ids[cm_indices]
#     nn_ms = get_unique_matches(cm_ids, r_mask)
#
#     return nn_ms
#
#
# def nearest_neighbor_thresh_match_score(des1, des2, t, s, dot_des, r_mask):
#     """
#     :param des1: N x C x H x W
#     :param des2: N x C x H x W
#     :param t: distance threshold
#     :param s: N x H x W x H x W. Matrix of correspondences
#     :param dot_des: N x H x W x H x W. Matrix of distances
#     :param r_mask: N x 1 x H x W
#     """
#     n, c, h, w = des1.size()
#     flat = h * w
#
#     des1 = des1.view(n, c, flat)
#     des2 = des2.view(n, c, flat)
#
#     ids = collect_ids(dot_des, 1)
#     cm_indices = get_correct_matches_indices(ids, s, r_mask)
#     cm_ids = ids[cm_indices]
#
#     # Test correct matches by a threshold
#
#     cm_des1 = des1[cm_ids[:, 0], :, cm_ids[:, 1]]
#     cm_des2 = des2[cm_ids[:, 0], :, cm_ids[:, 2]]
#
#     dist = torch.norm(cm_des1 - cm_des2, p=2, dim=1)
#     cm_indices = dist.lt(t).nonzero().squeeze(-1)
#
#     cm_ids = cm_ids[cm_indices]
#     nnt_ms = get_unique_matches(cm_ids, r_mask)
#
#     return nnt_ms
#
#
# def nearest_neighbor_ratio_match_score(des1, des2, rt, s, dot_des, r_mask):
#     """
#    :param des1: N x C x H x W
#    :param des2: N x C x H x W
#    :param rt: ratio threshold
#    :param s: N x H x W x H x W. Matrix of correspondences
#    :param dot_des: N x H x W x H x W. Matrix of distances
#    :param r_mask: N x 1 x H x W
#    """
#     n, c, h, w = des1.size()
#     flat = h * w
#
#     des1 = des1.view(n, c, flat)
#     des2 = des2.view(n, c, flat)
#
#     # Collect first and second closest matches
#     ids = collect_ids(dot_des, 2)
#     # Determine which of first matches are correct
#     cm_indices = get_correct_matches_indices(ids[::2], s, r_mask)
#     # Add their corresponding second matches
#     i = cm_indices.shape[0]
#     cm_indices = cm_indices.repeat(2)
#     cm_indices[i:] += 1
#
#     cm_ids = ids[cm_indices]
#
#     cm_des1 = des1[cm_ids[::2, 0], :, cm_ids[::2, 1]]
#     cm_des2b = des2[cm_ids[::2, 0], :, cm_ids[::2, 2]]
#     cm_des2c = des2[cm_ids[1::2, 0], :, cm_ids[1::2, 2]]
#
#     dist_b = torch.norm(cm_des1 - cm_des2b, p=2, dim=1)
#     dist_c = torch.norm(cm_des1 - cm_des2c, p=2, dim=1)
#
#     r_dist = dist_b / dist_c
#     cm_indices = r_dist.lt(rt).nonzero().squeeze(-1)
#
#     cm_ids = cm_ids[cm_indices]
#     nnr_ms = get_unique_matches(cm_ids, r_mask)
#
#     return nnr_ms

# vgg_structure = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128]
#
#
# def make_vgg_block(in_channels, out_channels, kernel_size, padding, activation):
#     block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
#     block += [nn.BatchNorm2d(out_channels)]
#
#     if activation is not None:
#         block += [activation]
#
#     return block
#
#
# def make_vgg_backbone():
#     layers = []
#     in_channels = 1
#     for v in vgg_structure:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             layers += make_vgg_block(in_channels, v, 3, 1, nn.ReLU(inplace=True))
#             in_channels = v
#
#     return nn.Sequential(*layers)
#
#
# def make_vgg_detector_head(grid_size):
#     layers = []
#     layers += make_vgg_block(vgg_structure[-1], 256, 3, 1, nn.ReLU(inplace=True))
#     layers += make_vgg_block(256, 1 + pow(grid_size, 2), 1, 0, None)
#
#     return nn.Sequential(*layers)
#
#
# def make_vgg_descriptor_head(descriptor_size):
#     layers = []
#     layers += make_vgg_block(vgg_structure[-1], 256, 3, 1, nn.ReLU(inplace=True))
#     layers += make_vgg_block(256, descriptor_size, 1, 0, None)
#
#     return nn.Sequential(*layers)

# class HomoHingeLoss(nn.Module):
#
#     def __init__(self, grid_size, pos_lambda, pos_margin, neg_margin):
#         super().__init__()
#         self.grid_size = grid_size
#
#         self.pos_lambda = pos_lambda
#
#         self.pos_margin = pos_margin
#         self.neg_margin = neg_margin
#
#     def forward(self, desc1, desc2, homo21, vis_mask1):
#         """
#         :param desc1: N x C x Hr x Wr
#         :param desc2: N x C x Hr x Wr
#         :param homo21: N x 3 x 3
#         :param vis_mask1: Mask of the first image. N x 1 x H x W
#         Note: 'r' suffix means reduced in 'grid_size' times
#         """
#         # We need to account for difference in size between descriptor and image for homography to work
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
#         loss = loss.sum() / norm
#
#         return loss