# import os
#
# import torch
#
# from ignite.contrib.handlers.custom_events import Events
#
# import Net.source.core.experiment as exp
# import Net.source.utils.eval_utils as ev
# import Net.source.datasets.dataset_utils as d
# import Net.source.core.loop as l
# import Net.source.nn.net.utils.model_utils as a
#
# from Net.source.nn.super_point.model import SuperPointNet
#
# from Net.source.core.experiment import Experiment
# from Net.source.loops.super_point import SuperPointLoop
#
# from Net.source.utils.matching_utils import DescriptorDistance
#
# from Net.source.core.ignite_metrics import AveragePeriodicMetric, DetailedMetric
# from Net.source.utils.ignite_utils import RepTransformer, MSTransformer, MMATransformer, \
#     TimeTransformer, PoseTransformer
#
# from Net.source.utils.eval_utils import print_summary, save_test_log, \
#     save_aachen_inference, join_logs
#
#
# class SuperPointExperiment(Experiment):
#
#     def get_main_loops(self):
#         if self.mode == l.TEST:
#             test_loop = SuperPointLoop.from_experiment(self, self.mode)
#
#             datasets = self.datasets_config[self.mode].keys()
#             if d.MEGADEPTH in datasets:
#                 px_thresh = self.metric_config[l.TEST][exp.PX_THRESH]
#                 ep_thresh = self.metric_config[l.TEST][exp.EP_THRESH]
#
#                 DetailedMetric(PoseTransformer(px_thresh, ep_thresh, DescriptorDistance.INV_COS_SIM, gt=False),
#                                len(px_thresh)) \
#                     .attach(test_loop.engine, ev.REL_POSE)
#                 DetailedMetric(RepTransformer(px_thresh, True), len(px_thresh)).attach(test_loop.engine, ev.REP)
#                 DetailedMetric(MSTransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), len(px_thresh)) \
#                     .attach(test_loop.engine, ev.MS)
#                 DetailedMetric(MMATransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), len(px_thresh)) \
#                     .attach(test_loop.engine, ev.MMA)
#
#                 @test_loop.engine.on(Events.EPOCH_COMPLETED)
#                 def on_epoch(engine):
#                     logs = join_logs(engine)
#
#                     print_summary(logs, self.metric_config[l.TEST])
#                     save_test_log(self.log_dir, logs, self.metric_config[l.TEST], self.model_config, datasets)
#
#             elif d.AACHEN in datasets:
#                 @test_loop.engine.on(Events.ITERATION_COMPLETED)
#                 def on_iteration_completed(engine):
#                     save_aachen_inference(self.datasets_config[d.AACHEN][d.DATASET_ROOT], engine.state.output)
#
#             elif d.HPATCHES_ILLUM in datasets or d.HPATCHES_VIEW in datasets:
#                 px_thresh = self.metric_config[l.TEST][exp.PX_THRESH]
#
#                 DetailedMetric(RepTransformer(px_thresh, True), len(px_thresh)).attach(test_loop.engine, ev.REP)
#                 DetailedMetric(MSTransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), len(px_thresh)) \
#                     .attach(test_loop.engine, ev.MS)
#                 DetailedMetric(MMATransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), len(px_thresh)) \
#                     .attach(test_loop.engine, ev.MMA)
#                 AveragePeriodicMetric(TimeTransformer()).attach(test_loop.engine, a.FORWARD_TIME)
#
#                 @test_loop.engine.on(Events.EPOCH_COMPLETED)
#                 def on_epoch(engine):
#                     logs = join_logs(engine)
#
#                     print_summary(logs, self.metric_config[l.TEST])
#                     print(f"Average running time: {engine.state.metrics[a.FORWARD_TIME]}")
#                     save_test_log(self.log_dir, logs, self.metric_config[l.TEST], self.model_config, datasets)
#
#             return [test_loop]
#
#         elif self.mode == l.ANALYZE:
#             analyze_loop = SuperPointLoop.from_experiment(self, self.mode)
#             return [analyze_loop]
#
#     def get_models(self):
#         model = SuperPointNet(self.model_config[exp.GRID_SIZE]).to(self.device)
#         return [model]
#
#     def load_checkpoints(self):
#         if exp.CHECKPOINT_NAME in self.model_config:
#             checkpoint_name = self.model_config[exp.CHECKPOINT_NAME][0]
#
#             model_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")
#             self.model[0].load_state_dict(torch.load(model_path, map_location=self.device))
#
#             print(f"Model {checkpoint_name} loaded")
#
#
#     def get_criterions(self):
#         pass
#
#     def get_optimizers(self):
#         pass
#
#     def bind_checkpoints(self):
#         pass
#
#     def start_logging(self):
#         pass
#
#     def end_logging(self):
#         pass