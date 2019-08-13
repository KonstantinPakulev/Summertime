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
#
# import os
# import sys
# import datetime
# from argparse import ArgumentParser
#
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
#
# import torch
# from torch.optim import Adam
# from torch.utils.data.dataloader import DataLoader
# from torchvision import transforms
#
# from tensorboardX import SummaryWriter
# from ignite.engine import Engine, Events
# from ignite.handlers import ModelCheckpoint, Timer
#
# from Net.source.nn.model import NetVGG
# from Net.source.nn.criterion import MSELoss, HardTripletLoss
# from Net.source.hpatches_dataset import (
#     HPatchesDataset,
#
#     TRAIN,
#     VALIDATE,
#     VALIDATE_SHOW,
#
#     IMAGE1,
#     IMAGE2,
#     HOMO12,
#     HOMO21,
#     S_IMAGE1,
#     S_IMAGE2,
#
#     Grayscale,
#     Normalize,
#     RandomCrop,
#     Rescale,
#     ToTensor
# )
#
# from Net.utils.common_utils import print_dict
# from Net.utils.eval_utils import (LOSS,
#                                   DET_LOSS,
#                                   DES_LOSS,
#                                   REP_SCORE,
#                                   MATCH_SCORE,
#                                   NN_MATCH_SCORE,
#                                   NNT_MATCH_SCORE,
#                                   NNR_MATCH_SCORE,
#                                   SHOW,
#
#                                   KP1,
#                                   KP2,
#                                   W_KP1,
#                                   W_KP2,
#                                   KP1_DESC,
#                                   KP2_DESC,
#
#                                   IMAGE_SIZE,
#                                   TOP_K,
#                                   KP_MT,
#                                   DS_MT,
#                                   DS_MR,
#
#                                   l_loss,
#                                   l_det_loss,
#                                   l_des_loss,
#                                   l_rep_score,
#                                   l_match_score,
#                                   l_collect_show,
#
#                                   plot_keypoints)
# from Net.utils.ignite_utils import AverageMetric, CollectMetric, AverageListMetric
# from Net.utils.image_utils import select_keypoints, warp_keypoints
# from Net.source.utils.model_utils import sample_descriptors
#
#
# def h_patches_dataset(mode, config):
#     transform = [Grayscale(),
#                  Normalize(mean=config.DATASET.view.MEAN, std=config.DATASET.view.STD),
#                  Rescale((960, 1280))]
#
#     if mode == TRAIN:
#         csv_file = config.DATASET.view.train_csv
#         transform += [RandomCrop((720, 960)),
#                       Rescale((240, 320))]
#         include_originals = False
#
#     elif mode == VALIDATE:
#         csv_file = config.DATASET.view.val_csv
#         transform += [Rescale((240, 320))]
#         include_originals = False
#     else:
#         csv_file = config.DATASET.view.show_csv
#         transform += [Rescale((320, 640))]
#         include_originals = True
#
#     transform += [ToTensor()]
#
#     return HPatchesDataset(root_path=config.DATASET.view.root,
#                            csv_file=csv_file,
#                            transform=transforms.Compose(transform),
#                            include_sources=include_originals)
#
#
# def attach_metrics(engine, config):
#     AverageMetric(l_loss, config.TRAIN.LOG_INTERVAL).attach(engine, LOSS)
#     AverageMetric(l_det_loss, config.TRAIN.LOG_INTERVAL).attach(engine, DET_LOSS)
#     AverageMetric(l_des_loss, config.TRAIN.LOG_INTERVAL).attach(engine, DES_LOSS)
#     AverageMetric(l_rep_score, config.TRAIN.LOG_INTERVAL).attach(engine, REP_SCORE)
#     AverageListMetric(l_match_score, config.TRAIN.LOG_INTERVAL).attach(engine, MATCH_SCORE)
#
#
# def output_metrics(writer, data_engine, state_engine, tag):
#     """
#     :param writer: SummaryWriter
#     :param data_engine: Engine to take data from
#     :param state_engine: Engine to take current state from
#     :param tag: Category to write data to
#     """
#     writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)
#     writer.add_scalar(f"{tag}/{DET_LOSS}", data_engine.state.metrics[DET_LOSS], state_engine.state.iteration)
#     writer.add_scalar(f"{tag}/{DES_LOSS}", data_engine.state.metrics[DES_LOSS], state_engine.state.iteration)
#     writer.add_scalar(f"{tag}/{REP_SCORE}", data_engine.state.metrics[REP_SCORE], state_engine.state.iteration)
#
#     t_ms, nn_ms, nnt_ms, nnr_ms = data_engine.state.metrics[MATCH_SCORE]
#     writer.add_scalar(f"{tag}/{MATCH_SCORE}", t_ms, state_engine.state.iteration)
#     writer.add_scalar(f"{tag}/{NN_MATCH_SCORE}", nn_ms, state_engine.state.iteration)
#     writer.add_scalar(f"{tag}/{NNT_MATCH_SCORE}", nnt_ms, state_engine.state.iteration)
#     writer.add_scalar(f"{tag}/{NNR_MATCH_SCORE}", nnr_ms, state_engine.state.iteration)
#
#
# def prepare_output_dict(batch, endpoint, config):
#     return {
#         LOSS: endpoint[LOSS],
#         DET_LOSS: endpoint[DET_LOSS],
#         DES_LOSS: endpoint[DES_LOSS],
#
#         KP1: endpoint[KP1],
#         KP2: endpoint[KP2],
#
#         W_KP1: endpoint[W_KP1],
#         W_KP2: endpoint[W_KP2],
#
#         KP1_DESC: endpoint[KP1_DESC],
#         KP2_DESC: endpoint[KP2_DESC],
#
#         IMAGE_SIZE: batch[IMAGE1].size(),
#
#         TOP_K: config.LOSS.TOP_K,
#
#         KP_MT: config.METRIC.DET_THRESH,
#         DS_MT: config.METRIC.DES_THRESH,
#         DS_MR: config.METRIC.DES_RATIO,
#     }
#
#
# def inference(model, batch, device, config):
#     image1, image2, homo12, homo21 = (
#         batch[IMAGE1].to(device),
#         batch[IMAGE2].to(device),
#         batch[HOMO12].to(device),
#         batch[HOMO21].to(device)
#     )
#
#     score1, desc1 = model(image1)
#     score2, desc2 = model(image2)
#
#     _, kp1 = select_keypoints(score1, config.LOSS.NMS_THRESH, config.LOSS.NMS_K_SIZE, config.LOSS.TOP_K)
#     _, kp2 = select_keypoints(score2, config.LOSS.NMS_THRESH, config.LOSS.NMS_K_SIZE, config.LOSS.TOP_K)
#
#     kp1_desc = sample_descriptors(desc1, kp1, config.MODEL.GRID_SIZE)
#     kp2_desc = sample_descriptors(desc2, kp2, config.MODEL.GRID_SIZE)
#
#     w_kp1 = warp_keypoints(kp1, homo12)
#     w_kp2 = warp_keypoints(kp2, homo21)
#
#     return {
#         S_IMAGE1: batch[S_IMAGE1],
#         S_IMAGE2: batch[S_IMAGE2],
#
#         KP1: kp1,
#         KP2: kp2,
#
#         W_KP1: w_kp1,
#         W_KP2: w_kp2,
#
#         KP1_DESC: kp1_desc,
#         KP2_DESC: kp2_desc,
#
#         TOP_K: config.LOSS.TOP_K,
#
#         KP_MT: config.METRIC.DET_THRESH,
#     }
#
#
# def train(config, device, num_workers, log_dir, checkpoint_dir):
#     """
#     :param config: config to use
#     :param device: cpu or gpu
#     :param num_workers: number of workers to load the data
#     :param log_dir: path to the directory to store tensorboard db
#     :param checkpoint_dir: path to the directory to save checkpoints
#     """
#     """
#     Dataset and data loaders preparation.
#     """
#
#     train_loader = DataLoader(h_patches_dataset(TRAIN, config),
#                               batch_size=config.TRAIN.BATCH_SIZE,
#                               shuffle=True,
#                               num_workers=num_workers)
#
#     val_loader = DataLoader(h_patches_dataset(VALIDATE, config),
#                             batch_size=config.VAL.BATCH_SIZE,
#                             shuffle=True,
#                             num_workers=num_workers)
#
#     show_loader = DataLoader(h_patches_dataset(VALIDATE_SHOW, config),
#                              batch_size=config.VAL_SHOW.BATCH_SIZE)
#
#     """
#     Model, optimizer and criterion settings.
#     Training and validation steps.
#     """
#
#     model = NetVGG(config.MODEL.GRID_SIZE, config.MODEL.DESCRIPTOR_SIZE).to(device)
#
#     mse_criterion = MSELoss(config.LOSS.NMS_THRESH, config.LOSS.NMS_K_SIZE,
#                             config.LOSS.TOP_K,
#                             config.LOSS.GAUSS_K_SIZE, config.LOSS.GAUSS_SIGMA, config.LOSS.DET_LAMBDA)
#
#     triplet_criterion = HardTripletLoss(config.MODEL.GRID_SIZE, config.LOSS.MARGIN, config.LOSS.DES_LAMBDA)
#
#     optimizer = Adam(model.parameters(), lr=config.TRAIN.LR)
#
#     def iteration(engine, batch):
#         image1, image2, homo12, homo21 = (
#             batch[IMAGE1].to(device),
#             batch[IMAGE2].to(device),
#             batch[HOMO12].to(device),
#             batch[HOMO21].to(device)
#         )
#
#         score1, desc1 = model(image1)
#         score2, desc2 = model(image2)
#
#         det_loss1, kp1 = mse_criterion(score1, score2, homo12)
#         det_loss2, kp2 = mse_criterion(score2, score1, homo21)
#
#         kp1_desc = sample_descriptors(desc1, kp1, config.MODEL.GRID_SIZE)
#         kp2_desc = sample_descriptors(desc2, kp2, config.MODEL.GRID_SIZE)
#
#         w_kp1 = warp_keypoints(kp1, homo12)
#         w_kp2 = warp_keypoints(kp2, homo21)
#
#         des_loss1 = triplet_criterion(w_kp1, kp1_desc, desc2)
#         des_loss2 = triplet_criterion(w_kp2, kp2_desc, desc1)
#
#         det_loss = (det_loss1 + det_loss2) / 2
#         des_loss = (des_loss1 + des_loss2) / 2
#
#         loss = det_loss + des_loss
#
#         return {
#             LOSS: loss,
#             DET_LOSS: det_loss,
#             DES_LOSS: des_loss,
#
#             KP1: kp1,
#             KP2: kp2,
#
#             W_KP1: w_kp1,
#             W_KP2: w_kp2,
#
#             KP1_DESC: kp1_desc,
#             KP2_DESC: kp2_desc
#         }
#
#     def train_iteration(engine, batch):
#         model.train()
#
#         with torch.autograd.set_detect_anomaly(True):
#             endpoint = iteration(engine, batch)
#
#             optimizer.zero_grad()
#             endpoint[LOSS].backward()
#             optimizer.step()
#
#         return prepare_output_dict(batch, endpoint, config)
#
#     trainer = Engine(train_iteration)
#
#     def validation_iteration(engine, batch):
#         model.eval()
#
#         with torch.no_grad():
#             endpoint = iteration(engine, batch)
#
#         return prepare_output_dict(batch, endpoint, config)
#
#     validator = Engine(validation_iteration)
#
#     def show_iteration(engine, batch):
#         model.eval()
#
#         with torch.no_grad():
#             endpoint = inference(model, batch, device, config)
#
#         return endpoint
#
#     show = Engine(show_iteration)
#     checkpoint_saver = ModelCheckpoint(checkpoint_dir, "my", save_interval=1, n_saved=3)
#     trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, {'model': model})
#
#     """
#     Visualisation utils, logging and metrics
#     """
#
#     writer = SummaryWriter(log_dir=log_dir)
#
#     attach_metrics(trainer, config)
#     attach_metrics(validator, config)
#     CollectMetric(l_collect_show).attach(show, SHOW)
#
#     """
#     Registering callbacks for sending summary to tensorboard
#     """
#     epoch_timer = Timer(average=False)
#     batch_timer = Timer(average=True)
#
#     @trainer.on(Events.ITERATION_COMPLETED)
#     def on_iteration_completed(engine):
#         if engine.state.iteration % config.TRAIN.LOG_INTERVAL == 0:
#             output_metrics(writer, engine, engine, "train")
#
#         if engine.state.iteration % config.VAL.LOG_INTERVAL == 0:
#             validator.run(val_loader)
#             output_metrics(writer, validator, engine, "val")
#
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def on_epoch_completed(engine):
#         if engine.state.epoch % config.VAL_SHOW.LOG_INTERVAL == 0:
#             show.run(show_loader)
#             plot_keypoints(writer, engine.state.epoch, show.state.metrics[SHOW])
#
#         # validator.run(val_loader)
#         text = f"""
#                 Epoch {engine.state.epoch} completed.
#                 \tFinished in {datetime.timedelta(seconds=epoch_timer.value())}.
#                 \tAverage time per batch is {batch_timer.value():.2f} seconds
#                 \tDetector learning rate is: {optimizer.param_groups[0]["lr"]}
#                 """
#         writer.add_text("Log", text, engine.state.epoch)
#
#     epoch_timer.attach(trainer,
#                        start=Events.EPOCH_STARTED,
#                        resume=Events.ITERATION_STARTED,
#                        pause=Events.ITERATION_COMPLETED)
#     batch_timer.attach(trainer,
#                        start=Events.EPOCH_STARTED,
#                        resume=Events.ITERATION_STARTED,
#                        pause=Events.ITERATION_COMPLETED,
#                        step=Events.ITERATION_COMPLETED)
#
#     trainer.run(train_loader, max_epochs=config.TRAIN.NUM_EPOCHS)
#
#     writer.close()
#
#
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--log_dir", type=str)
#     parser.add_argument("--checkpoint_dir", type=str)
#     parser.add_argument("--config", type=str)
#     parser.add_argument("--load_path", type=str)
#
#     args = parser.parse_args()
#
#     from legacy.train_config import cfg as train_config
#     from legacy.test_config import cfg as test_config
#
#     if args.config == 'test':
#         _config = test_config
#     else:
#         _config = train_config
#
#     print_dict(_config)
#
#     _device, _num_workers = (torch.device('cuda'), 8) if torch.cuda.is_available() else (torch.device('cpu'), 0)
#     _log_dir = args.log_dir
#     _checkpoint_dir = args.checkpoint_dir
#
#     train(_config, _device, _num_workers, _log_dir, _checkpoint_dir)


