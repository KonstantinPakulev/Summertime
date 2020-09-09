import os

import torch
from torch.optim import Adam


from ignite.contrib.handlers.custom_events import Events, CustomPeriodicEvent

import Net.source.core.loop as l
import Net.source.core.experiment as exp
import Net.source.nn.net.utils.endpoint_utils as eu
import Net.source.datasets.dataset_utils as du
import Net.source.utils.metric_utils as meu

from Net.source.core.experiment import Experiment
from Net.source.core.loop import Loop
from Net.source.datasets.dataset import create_dataset, create_loader

from Net.source.core.wrapper import CriterionChain, OptimizerChain

# Model imports

from Net.source.nn.net.models.backbone_models import VGGJointBackbone

from Net.source.nn.net.model_wrappers import NetContainer, BackboneWrapper, DetJointBranchWrapper,\
    DescJointBranchWrapper, LocJointBranchWrapper
from Net.source.nn.net.criterion_wrappers import DescTripletLossWrapper, LocEpipolarLossWrapper,\
    DetConfLossWrapper, DetJointLossWrapper, LocPoseLossWrapper

from Net.source.utils.matching_utils import DescriptorDistance

from Net.source.utils.log_utils import plot_losses_tensorboard, plot_metrics_tensorboard, \
    plot_scores, plot_kp_matches, plot_desc_matches, join_logs, print_summary, test_log_to_csv, save_aachen_inference


class NetVGGExperiment(Experiment):

    def get_model(self):
        if self.model_version in ['v1', 'v2', 'v3', 'v4']:
            model = NetContainer(self.device, [BackboneWrapper(self.device, self.model_config,
                                                               VGGJointBackbone.from_config(self.model_config)),
                                               DetJointBranchWrapper(self.device, self.model_config),
                                               LocJointBranchWrapper(self.device, self.model_config),
                                               DescJointBranchWrapper(self.device, self.model_config)])

        else:
            model = NetContainer(self.device, [BackboneWrapper(self.device, self.model_config,
                                                               VGGJointBackbone.from_config(self.model_config)),
                                               DetJointBranchWrapper(self.device, self.model_config),
                                               DescJointBranchWrapper(self.device, self.model_config)])

        return model.to(self.device)

    def get_criterion_chain(self):
        if self.model_version in ['v2', 'v3', 'v4']:
            criterion_chain = CriterionChain([DescTripletLossWrapper(self.device, self.model_config, self.criterion_config),
                                              DetConfLossWrapper(self.device, self.criterion_config),
                                              DetJointLossWrapper(self.device, self.criterion_config),
                                              LocEpipolarLossWrapper(self.device, self.criterion_config)])

        elif self.model_version == 'v1':
            criterion_chain = CriterionChain([DescTripletLossWrapper(self.device, self.model_config, self.criterion_config),
                                              DetConfLossWrapper(self.device, self.criterion_config),
                                              DetJointLossWrapper(self.device, self.criterion_config),
                                              LocEpipolarLossWrapper(self.device, self.criterion_config),
                                              LocPoseLossWrapper(self.device, self.criterion_config)])

        else:
            criterion_chain = CriterionChain([DescTripletLossWrapper(self.device, self.model_config, self.criterion_config),
                                              DetConfLossWrapper(self.device, self.criterion_config),
                                              DetJointLossWrapper(self.device, self.criterion_config)])

        return criterion_chain

    def get_optimizer_chain(self):
        optimizer = Adam(self.model.parameters(), self.experiment_config[exp.LR])
        optimizer_chain = OptimizerChain(optimizer)

        return optimizer_chain

    def get_loops(self):
        if self.mode == l.TRAIN:
            train_loop = get_loop(self, self.mode)
            val_loop = get_loop(self, l.VAL)
            visualize_loop = get_loop(self, l.VISUALIZE)

            train_config = self.metric_config[l.TRAIN]
            train_config[l.MODE] = l.TRAIN

            val_config = self.metric_config[l.VAL]
            val_config[l.MODE] = l.VAL

            visualize_config = self.metric_config[l.VISUALIZE]

            self.criterion_chain.attach(train_loop.engine, train_config)
            self.criterion_chain.attach(val_loop.engine, val_config)

            self.model.attach(train_loop.engine, train_config)
            self.model.attach(val_loop.engine, val_config)

            # TODO. Add to the heart of the container.  Calculate only during test with measure_time flag
            # AveragePeriodicMetric(TimeTransformer()).attach(engine, n.FORWARD_TIME)

            train_loss_event = CustomPeriodicEvent(n_iterations=train_config[exp.LOSS_LOG_ITER])
            train_metric_event = CustomPeriodicEvent(n_iterations=train_config[exp.METRIC_LOG_ITER])
            val_event = CustomPeriodicEvent(n_iterations=val_config[exp.LOG_ITER])
            visualize_event = CustomPeriodicEvent(n_epochs=visualize_config[exp.LOG_EPOCH])

            train_loss_event.attach(train_loop.engine)
            train_metric_event.attach(train_loop.engine)
            val_event.attach(train_loop.engine)
            visualize_event.attach(train_loop.engine)

            @train_loop.engine.on(train_loss_event._periodic_event_completed)
            def on_train_loss(engine):
                plot_losses_tensorboard(self.writer, engine, engine, l.TRAIN)

            @train_loop.engine.on(train_metric_event._periodic_event_completed)
            def on_train_metric(engine):
                plot_metrics_tensorboard(self.writer, engine, engine, l.TRAIN)

            @train_loop.engine.on(val_event._periodic_event_completed)
            def on_val(engine):
                val_loop.run(1)

                plot_losses_tensorboard(self.writer, val_loop.engine, engine, l.VAL)
                plot_metrics_tensorboard(self.writer, val_loop.engine, engine, l.VAL)

            @train_loop.engine.on(visualize_event._periodic_event_completed)
            def on_visualize(engine):
                visualize_loop.run(1)

                plot_scores(self.writer, engine, visualize_loop.engine, (eu.SCORE1, eu.SCORE2))
                plot_scores(self.writer, engine, visualize_loop.engine, (eu.SAL_SCORE1, eu.SAL_SCORE2))
                plot_scores(self.writer, engine, visualize_loop.engine, (eu.CONF_SCORE1, eu.CONF_SCORE2), True)

                plot_kp_matches(self.writer, engine, visualize_loop.engine, visualize_config[exp.PX_THRESH])
                plot_desc_matches(self.writer, engine, visualize_loop.engine, visualize_config[exp.PX_THRESH], DescriptorDistance.INV_COS_SIM)

            return [train_loop, val_loop, visualize_loop]

        elif self.mode == l.TEST:
            test_loop = get_loop(self, self.mode)

            test_config = self.metric_config[l.TEST]
            test_config[l.MODE] = l.TEST
            test_config[du.DATASET_NAME] = list(self.dataset_config[self.mode].keys())

            self.model.attach(test_loop.engine, test_config)

            if du.AACHEN in test_config[du.DATASET_NAME]:
                @test_loop.engine.on(Events.ITERATION_COMPLETED)
                def on_iteration_completed(engine):
                    save_aachen_inference(self.dataset_config[self.mode], engine.state.output)

            else:
                @test_loop.engine.on(Events.EPOCH_COMPLETED)
                def on_epoch(engine):
                    logs = join_logs(engine)

                    print_summary(logs, test_config)
                    test_log_to_csv(self.log_dir, logs, test_config, self.model_config, test_config[du.DATASET_NAME])

            return [test_loop]

        elif self.mode == l.ANALYZE:
            analyze_loop = get_loop(self, self.mode)

            return [analyze_loop]

        else:
            return None

    def load_checkpoints(self):
        checkpoint_name = self.model_config.get(exp.CHECKPOINT_NAME)

        if checkpoint_name is not None:
            model_path = os.path.join(self.checkpoint_dir, f"{exp.CHECKPOINT_PREFIX}_{exp.MODEL}_{checkpoint_name}.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

            print(f"Model {checkpoint_name} is loaded")

            if self.optimizer_chain is not None:
                optimizer_path = os.path.join(self.checkpoint_dir,
                                              f"{exp.CHECKPOINT_PREFIX}_{exp.OPTIMIZER}_{checkpoint_name}.pth")
                self.optimizer_chain.load_state_dict(torch.load(optimizer_path, map_location=self.device))

                print(f"Optimizer {checkpoint_name} is loaded")


    def get_checkpoint_params(self):
        return meu.EMS, lambda engine: engine.state.metrics[meu.EMS][0]


"""
Support utils
"""


def get_loop(experiment, mode):
    dataset = create_dataset(experiment.dataset_config[mode], experiment.model_config, mode)
    loader = create_loader(dataset, experiment.loader_config[mode])

    return Loop(experiment.device, mode, experiment.model,
                experiment.criterion_chain, experiment.optimizer_chain, loader)


# Legacy code

#         @test_loop.engine.on(Events.EPOCH_COMPLETED)
#         def on_epoch(engine):
#             logs = join_logs(engine, [ev.REP, ev.MS, ev.MMA])
#
#             print_summary(logs, self.metric_config[l.TEST])
#             print(f"Average running time: {engine.state.metrics[a.FORWARD_TIME]}")
#             save_test_log(self.log_dir, logs, self.metric_config[l.TEST], self.models_config, datasets)

#
#     DetailedMetric(DescriptorAnalysisTransformer(self.models_config[exp.GRID_SIZE]), 1).attach(analyze_loop.engine, an.ANALYSIS_DATA)
#
#     @analyze_loop.engine.on(Events.EPOCH_COMPLETED)
#     def on_epoch(engine):
#         save_analysis_log(self.log_dir, engine.state.metrics[an.ANALYSIS_DATA], self.models_config)

# if self.model_version in ['v10']:
#     model = NetContainer([BackboneWrapper(VGGMSDBackbone()),
#                           DetMSBranchWrapper(DetMSDBranch.from_config(self.model_config), self.model_config),
#                           LocMSBranchWrapper(self.model_config),
#                           DescMSBranchWrapper(self.model_config)])
# if self.model_version == 'v10':
#     criterion_chain = CriterionChain([DetMSLossWrapper(self.criterion_config),
#                                       DescTripletLossWrapper(self.model_config, self.criterion_config),
#                                       LocEpipolarLossWrapper(self.criterion_config),
#                                       LocPoseLossWrapper(self.criterion_config)])
