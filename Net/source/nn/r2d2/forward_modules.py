# import torch
#
# import Net.source.core.base_experiment as exp
# import Net.source.nn.net.utils.endpoint_utils as f
# import Net.source.datasets.dataset_utils as d
# import Net.source.core.base_loop as l
# import Net.source.nn.net.utils.model_utils as a
#
# from Net.source.nn.net.utils.endpoint_utils import select_kp_top_k, select_kp_by_thresh, localize_kp, sample_loc, \
#     sample_descriptors, warp_points
#
#
# class SIForward:
#
#     def __init__(self, model, mode, model_config, dataset_config):
#         self.model = model
#
#         self.mode = mode
#         self.model_config = model_config
#         self.dataset_config = dataset_config
#
#     def extract_keypoints(self, endpoint):
#         kernel_size = self.model_config[exp.NMS_KP_KERNEL_SIZE]
#
#         score = endpoint[f.SCORE]
#
#         if exp.TOP_K in self.model_config:
#             top_k = self.model_config[exp.TOP_K]
#
#             kp = select_kp_top_k(score, kernel_size, top_k)
#             kp = localize_kp(score, kp)
#
#         elif exp.SCORE_THRESH in self.model_config:
#             score_thresh = self.model_config[exp.SCORE_THRESH]
#
#             kp = select_kp_by_thresh(score, kernel_size, score_thresh)
#             kp = localize_kp(score, kp)
#
#         else:
#             kp = None
#
#         endpoint[f.KP] = kp
#
#         return endpoint
#
#     def run(self, image):
#         detailed = self.model_config[exp.DETAILED]
#
#         if self.mode == l.TEST and detailed and torch.cuda.is_available():
#             start = torch.cuda.Event(enable_timing=True)
#             end = torch.cuda.Event(enable_timing=True)
#
#             start.record()
#             endpoint = self.model(image)
#             end.record()
#
#             torch.cuda.synchronize()
#             forward_time = start.elapsed_time(end)
#
#             endpoint[f.MODEL_INFO][a.FORWARD_TIME] = forward_time
#
#         else:
#             endpoint = self.model(image)
#
#         endpoint = self.extract_keypoints(endpoint)
#         return endpoint
#
#     def __call__(self, engine, batch):
#         endpoint1 = rename_keys1(remove_output_keys(self.mode, self.run(batch.get(d.IMAGE1))))
#         endpoint2 = rename_keys2(remove_output_keys(self.mode, self.run(batch.get(d.IMAGE2))))
#
#         endpoint = {**endpoint1, **endpoint2}
#         endpoint = post_process_keypoints(endpoint, batch)
#
#         return endpoint
#
#
# class SDIForward(SIForward):
#
#     def process_descriptors(self, endpoint):
#         grid_size = self.model_config[exp.GRID_SIZE]
#
#         kp_desc = sample_descriptors(endpoint[f.DESC], endpoint[f.KP], grid_size)
#
#         endpoint[f.KP_DESC] = kp_desc
#
#         return endpoint
#
#     def run(self, image):
#         endpoint = super().run(image)
#         endpoint = self.process_descriptors(endpoint)
#         return endpoint
#
#     def __call__(self, engine, batch):
#         datasets = self.dataset_config.keys()
#
#         if d.AACHEN in datasets:
#             endpoint = rename_keys1(remove_output_keys(self.mode, self.run(batch.get(d.IMAGE1))))
#         else:
#             endpoint = super().__call__(engine, batch)
#
#         return endpoint
#
#
# class SDDIForward(SDIForward):
#
#     def extract_keypoints(self, endpoint):
#         kernel_size = self.model_config[exp.NMS_KP_KERNEL_SIZE]
#
#         score = endpoint[f.SCORE]
#
#         if exp.TOP_K in self.model_config:
#             top_k = self.model_config[exp.TOP_K]
#
#             kp = select_kp_top_k(score, kernel_size, top_k)
#             # kp = localize_kp(score, kp)
#
#         elif exp.SCORE_THRESH in self.model_config:
#             score_thresh = self.model_config[exp.SCORE_THRESH]
#
#             kp = select_kp_by_thresh(score, kernel_size, score_thresh)
#             # kp = localize_kp(score, kp)
#
#         else:
#             kp = None
#
#         endpoint[f.KP] = kp
#
#         return endpoint
#
#     @staticmethod
#     def process_deltas(endpoint):
#         kp_delta = sample_loc(endpoint[f.DELTA], endpoint[f.KP])
#
#         endpoint[f.KP] = endpoint[f.KP].float() + kp_delta
#
#         return endpoint
#
#     def run(self, image):
#         endpoint = super().run(image)
#         endpoint = self.process_deltas(endpoint)
#
#         return endpoint
#
#
# class SuperPointForward(SDIForward):
#
#     def extract_keypoints(self, endpoint):
#         kernel_size = self.model_config[exp.NMS_KP_KERNEL_SIZE]
#
#         score = endpoint[f.SCORE]
#
#         if exp.TOP_K in self.model_config:
#             top_k = self.model_config[exp.TOP_K]
#
#             kp = select_kp_top_k(score, kernel_size, top_k)
#
#         # elif exp.SCORE_THRESH in self.model_config:
#         #     score_thresh = self.model_config[exp.SCORE_THRESH]
#         #
#         #     kp = select_kp_thresh(score, kernel_size, score_thresh)
#
#         else:
#             kp = None
#
#         endpoint[f.KP] = kp
#
#         return endpoint
#
#
# class R2D2Forward(SDIForward):
#
#     def extract_keypoints(self, endpoint):
#         kernel_size = self.model_config[exp.NMS_KP_KERNEL_SIZE]
#
#         score = endpoint[f.SCORE]
#         disc = endpoint[f.DISC]
#
#         disc_mask = disc.ge(0.7)
#         score_mask = score.ge(0.7)
#         conf_score = score * disc * score_mask.float() * disc_mask.float()
#
#         if exp.TOP_K in self.model_config:
#             top_k = self.model_config[exp.TOP_K]
#
#             kp = select_kp_top_k(conf_score, kernel_size, top_k)
#
#         # elif exp.SCORE_THRESH in self.model_config:
#         #     score_thresh = self.model_config[exp.SCORE_THRESH]
#         #
#         #     kp = select_kp_thresh(conf_score, kernel_size, score_thresh)
#
#         else:
#             kp = None
#
#         endpoint[f.KP] = kp
#
#         return endpoint
#
#
# """
# Support utils
# """
#
#
# def remove_output_keys(mode, endpoint):
#     if mode in [l.TRAIN, l.VAL]:
#         output_keys = [f.MODEL_INFO]
#
#     elif mode == l.TEST:
#         output_keys = [f.DESC, f.DISC]
#     else:
#         output_keys = []
#
#     for _k in output_keys:
#         if _k in endpoint:
#             del endpoint[_k]
#
#     return endpoint
#
#
# def rename_keys1(endpoint):
#     mapping = {
#         f.SCORE: f.SCORE1,
#         f.DESC: f.DESC1,
#         f.DISC: f.DISC1,
#         f.DELTA: f.DELTA1,
#         f.MODEL_INFO: f.MODEL_INFO1,
#
#         f.KP: f.KP1,
#         f.KP_DESC: f.KP1_DESC
#     }
#     return rename_keys(endpoint, mapping)
#
#
# def rename_keys2(endpoint):
#     mapping = {
#         f.SCORE: f.SCORE2,
#         f.DESC: f.DESC2,
#         f.DISC: f.DISC2,
#         f.DELTA: f.DELTA2,
#         f.MODEL_INFO: f.MODEL_INFO2,
#
#         f.KP: f.KP2,
#         f.KP_DESC: f.KP2_DESC
#     }
#     return rename_keys(endpoint, mapping)
#
#
# def rename_keys(endpoint, mapping):
#     for _k, _v in mapping.items():
#         if _k in endpoint:
#             endpoint[_v] = endpoint.pop(_k)
#
#     return endpoint
#
#
# def post_process_keypoints(endpoint, batch):
#     kp1, kp2, score1, score2 = endpoint[f.KP1], endpoint[f.KP2], endpoint[f.SCORE1], endpoint[f.SCORE2]
#
#     w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask = warp_points(kp1, kp2, score1, score2, batch)
#
#     endpoint[f.W_KP1] = w_kp1
#     endpoint[f.W_KP2] = w_kp2
#
#     endpoint[f.W_VIS_KP1_MASK] = w_vis_kp1_mask
#     endpoint[f.W_VIS_KP2_MASK] = w_vis_kp2_mask
#
#     return endpoint
