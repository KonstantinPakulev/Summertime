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
#
# class DetectorConfig(MainConfig):
#
#     def init_checkpoint(self):
#         self.checkpoint.SAVE_INTERVAL = 100
#         self.checkpoint.N_SAVED = 3
#
#
# class SPConfig(MainConfig):
#
#     def init_criterion(self):
#         self.criterion.DES_LAMBDA = 1
#         self.criterion.POS_LAMBDA = 250
#         self.criterion.POS_MARGIN = 1.0
#         self.criterion.NEG_MARGIN = 0.2
#
#         self.criterion.DET_LAMBDA = 120
#
#         self.criterion.NMS_THRESH = 0.0
#         self.criterion.NMS_K_SIZE = 5
#
#         self.criterion.TOP_K = 512
#
#         self.criterion.GAUSS_K_SIZE = 15
#         self.criterion.GAUSS_SIGMA = 0.5

# def get_models(self, models_configs):

#
#
# def get_criterions(self, models_configs, criterions_configs):

#
#
# def get_optimizers(self, models, optimizers_configs):


# class DefaultExperiment(Experiment, ABC):
#
#     def __init__(self, device, log_dir=None, checkpoint_dir=None, checkpoint_iter=None):
#         super().__init__(device, log_dir, checkpoint_dir, checkpoint_iter)
#
#         self.writer = None
#
#     def load_checkpoint(self):

#
#     def start_logging(self):
#         self.writer = SummaryWriter(log_dir=self.log_dir)
#
#     def init_engines(self):
#         model = self.models[MODEL]
#
#         optimizer = self.optimizers[OPTIMIZER]
#
#         def train_iteration(engine, batch):
#             model.train()
#
#             with torch.autograd.set_detect_anomaly(True):
#                 endpoint = self.iteration(engine, batch)
#
#                 optimizer.zero_grad()
#                 endpoint[LOSS].backward()
#                 optimizer.step()
#
#             return endpoint
#
#         def val_iteration(engine, batch):
#             model.eval()
#
#             with torch.no_grad():
#                 endpoint = self.iteration(engine, batch)
#
#             return endpoint
#
#         def show_iteration(engine, batch):
#             model.eval()
#
#             with torch.no_grad():
#                 endpoint = self.inference(engine, batch)
#
#             return endpoint
#
#         self.engines[TRAIN_ENGINE] = Engine(train_iteration)
#         self.engines[VAL_ENGINE] = Engine(val_iteration)
#         self.engines[SHOW_ENGINE] = Engine(show_iteration)
#
#     def bind_checkpoint(self):

#
#     def run_experiment(self):
#         es = self.config.experiment
#
#         train_engine = self.engines[TRAIN_ENGINE]
#         train_loader = self.loaders[TRAIN_LOADER]
#
#         train_engine.run(train_loader, max_epochs=es.NUM_EPOCHS)
#
#     def stop_logging(self):
#         self.writer.close()
#
# class MainAlterConfig(MainConfig):
#
#     def init_criterion(self):
#         self.criterion.DES_LAMBDA = 1
#         self.criterion.MARGIN = 1
#         self.criterion.NUM_NEG = 1
#         self.criterion.SOS_NEG = 8
#
#         self.criterion.DET_LAMBDA = 120
#
#         self.criterion.NMS_THRESH = 0.0
#         self.criterion.NMS_K_SIZE = 5
#
#         self.criterion.TOP_K = 512
#
#         self.criterion.GAUSS_K_SIZE = 15
#         self.criterion.GAUSS_SIGMA = 0.5
#
#
# class DebugConfig(MainConfig):
#
#     def init_dataset(self):
#         self.dataset.DATASET_ROOT = "../data/hpatch_v_sequence"
#         self.dataset.DATASET_MEAN = 0.4230204841414801
#         self.dataset.DATASET_STD = 0.25000138349993173
#         self.dataset.TRAIN_CSV = "debug.csv"
#         self.dataset.VAL_CSV = "debug.csv"
#         self.dataset.SHOW_CSV = "debug.csv"
#         self.dataset.ANALYZE_CSV = "analyze.csv"
#
#     def init_loader(self):
#         self.loader.TRAIN_BATCH_SIZE = 1
#         self.loader.VAL_BATCH_SIZE = 1
#         self.loader.SHOW_BATCH_SIZE = 1
#         self.loader.ANALYZE_BATCH_SIZE = 1
#         self.loader.NUM_WORKERS = 0
#
#     def init_model(self):
#         self.model.GRID_SIZE = 8
#         self.model.DESCRIPTOR_SIZE = 4
#         self.model.NMS_KERNEL_SIZE = 15
#
#     def init_log(self):
#         self.log.TRAIN = EasyDict()
#         self.log.TRAIN.LOSS_LOG_INTERVAL = 1
#         self.log.TRAIN.METRIC_LOG_INTERVAL = 1
#
#         self.log.VAL = EasyDict()
#         self.log.VAL.LOG_INTERVAL = 1
#
#         self.log.SHOW = EasyDict()
#         self.log.SHOW.LOG_INTERVAL = 1
#
#     def init_experiment(self):
#         self.experiment.NUM_EPOCHS = 100
#
#     def init_checkpoint(self):
#         self.checkpoint.SAVE_INTERVAL = 19
#         self.checkpoint.N_SAVED = 5
# thresh_mask = dnn_values[:, :, 0].le(des_match_thresh)  # B x N
# nn_thresh_match_score = \
#     abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, thresh_mask, wv_kp2_mask, dnn_kp_values, dnn_ids[:, :, 0],
#                                           kp_match_thresh)
#
# dist_ratio = dnn_values[:, :, 0] / dnn_values[:, :, 1]
# ratio_mask = dist_ratio.le(des_match_ratio)  # B x N
# nn_ratio_match_score = \
#     abstract_nearest_neighbor_match_score(kp1, w_kp2, kp2, ratio_mask, wv_kp2_mask, dnn_kp_values, dnn_ids[:, :, 0],
#                                           kp_match_thresh)
# class TrainDetMetricBinder(BaseMetricBinder):
#
#     def bind(self, engines, loaders):
#         train_engine = engines[TRAIN_ENGINE]
#         val_engine = engines[VAL_ENGINE]
#         show_engine = engines[SHOW_ENGINE]
#
#         ls = self.config.log
#         ms = self.config.metric
#
#         def l_loss(x):
#             return x[LOSS]
#
#         def l_rep_score(x):
#             rep1 = repeatability_score(x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], ms.DET_THRESH)
#             rep2 = repeatability_score(x[KP2], x[W_KP1], x[KP1], x[WV_KP1_MASK], ms.DET_THRESH)
#             return (rep1 + rep2) / 2
#
#         def l_collect_show(x):
#             return x[S_IMAGE1][0], x[S_IMAGE2][0], x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], None, None
#
#         # Train metrics
#         AveragePeriodicMetric(l_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, LOSS)
#         PeriodicMetric(l_rep_score, ls.TRAIN.MAIN_METRIC_LOG_INTERVAL).attach(train_engine, REP_SCORE)
#
#         # Val metrics
#         AveragePeriodicMetric(l_loss).attach(val_engine, LOSS)
#         AveragePeriodicMetric(l_rep_score).attach(val_engine, REP_SCORE)
#
#         # Show metrics
#         CollectMetric(l_collect_show).attach(show_engine, SHOW)
#
#         train_loss_event = CustomPeriodicEvent(n_iterations=ls.TRAIN.LOSS_LOG_INTERVAL)
#         train_metric_event = CustomPeriodicEvent(n_iterations=ls.TRAIN.METRIC_LOG_INTERVAL)
#
#         val_event = CustomPeriodicEvent(n_iterations=ls.VAL.LOG_INTERVAL)
#         show_event = CustomPeriodicEvent(n_epochs=ls.SHOW.LOG_INTERVAL)
#
#         val_loader = loaders[VAL_LOADER]
#         show_loader = loaders[SHOW_LOADER]
#
#         # Attach events to train engine
#         train_loss_event.attach(train_engine)
#         train_metric_event.attach(train_engine)
#
#         val_event.attach(train_engine)
#         show_event.attach(train_engine)
#
#         # Train events
#         @train_engine.on(train_loss_event._periodic_event_completed)
#         def on_tle(engine):
#             output_loss(self.writer, train_engine, train_engine, "train")
#
#         @train_engine.on(train_metric_event._periodic_event_completed)
#         def on_tme(engine):
#             output_metric(self.writer, train_engine, train_engine, "train")
#
#         # Val event
#         @train_engine.on(val_event._periodic_event_completed)
#         def on_ve(engine):
#             val_engine.run(val_loader)
#
#             output_loss(self.writer, val_engine, train_engine, "val")
#             output_metric(self.writer, val_engine, train_engine, "val")
#
#         # Show event
#         @train_engine.on(show_event._periodic_event_completed)
#         def on_se(engine):
#             show_engine.run(show_loader)
#
#             plot_keypoints_and_descriptors(self.writer, train_engine.state.epoch, show_engine.state.metrics[SHOW],
#                                            ms.DET_THRESH)
# def output_metric(writer, data_engine, state_engine, tag):
#     writer.add_scalar(f"{tag}/{REP_SCORE}", data_engine.state.metrics[REP_SCORE], state_engine.state.iteration)
#
# def output_loss(writer, data_engine, state_engine, tag):
#     writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)