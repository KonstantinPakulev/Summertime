import os

import torch
from torch.optim import Adam

from Net.experiments.base import Experiment
from Net.experiments.config import *
from Net.experiments.loop import *
from Net.source.nn.model import NetVGG
from Net.source.nn.criterion import MSELoss, DenseInterQTripletLoss
from Net.source.utils.eval_utils import tb_plot_keypoints_and_descriptors, io_plot_keypoints_and_descriptors

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.custom_events import CustomPeriodicEvent

CHECKPOINT_PREFIX = "my"

MODEL = 'model'
OPTIMIZER = 'optimizer'

REP_SCORE = 'repeatability_score'
NN_MATCH_SCORE = 'nearest_neighbour_match_score'


def tb_output_losses(writer, data_engine, state_engine, tag):
    writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{DET_LOSS}", data_engine.state.metrics[DET_LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{DES_LOSS}", data_engine.state.metrics[DES_LOSS], state_engine.state.iteration)


def tb_output_metrics(writer, data_engine, state_engine, tag):
    rep, nn = data_engine.state.metrics[METRICS]
    writer.add_scalar(f"{tag}/{REP_SCORE}", rep, state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{NN_MATCH_SCORE}", nn, state_engine.state.iteration)


class TrainExperiment(Experiment):

    def __init__(self, device, log_dir=None, checkpoint_dir=None, checkpoint_iter=None):
        super().__init__(device, log_dir, checkpoint_dir, checkpoint_iter)

        self.writer = None

    def get_models_configs(self):
        return ModelConfig().get()

    def get_criterions_configs(self):
        return CriterionConfig().get()

    def get_experiments_configs(self):
        return TrainExperimentConfig().get()

    def get_models(self, models_configs):
        model = NetVGG(models_configs.GRID_SIZE,
                       models_configs.DESCRIPTOR_SIZE,
                       models_configs.NMS_KERNEL_SIZE).to(self.device)
        return model

    def get_criterions(self, models_configs, criterions_configs):
        det_criterion = MSELoss(criterions_configs.NMS_THRESH, criterions_configs.NMS_K_SIZE,
                                criterions_configs.TOP_K,
                                criterions_configs.GAUSS_K_SIZE, criterions_configs.GAUSS_SIGMA,
                                criterions_configs.DET_LAMBDA)
        des_criterion = DenseInterQTripletLoss(models_configs.GRID_SIZE, criterions_configs.MARGIN,
                                               criterions_configs.DES_LAMBDA)
        return det_criterion, des_criterion

    def get_optimizers(self, models):
        optimizer = Adam(models.parameters())
        return optimizer

    def load_checkpoints(self, models, optimizers):
        if self.checkpoint_iter is not None:
            model_path = os.path.join(self.checkpoint_dir,
                                      f"{CHECKPOINT_PREFIX}_{MODEL}_{self.checkpoint_iter}.pth")
            models.load_state_dict(torch.load(model_path, map_location=self.device))

            optimizer_path = os.path.join(self.checkpoint_dir,
                                          f"{CHECKPOINT_PREFIX}_{OPTIMIZER}_{self.checkpoint_iter}.pth")
            optimizers.load_state_dict(torch.load(optimizer_path, map_location=self.device))

    def get_train_dataset_config(self):
        return TrainDatasetConfig().get()

    def get_val_dataset_config(self):
        return ValDatasetConfig().get()

    def get_show_dataset_config(self):
        return ShowDatasetConfig().get()

    def get_train_loader_config(self):
        return TrainValLoaderConfig().get()

    def get_show_loader_config(self):
        return ShowLoaderConfig().get()

    def get_train_log_config(self):
        return TrainLogConfig().get()

    def get_val_log_config(self):
        return ValLogConfig().get()

    def get_show_log_config(self):
        return ShowLogConfig().get()

    def get_vsat_experiment_config(self):
        return VSATExperimentConfig().get()

    def get_loop(self):
        train_dataset_config = self.get_train_dataset_config()
        val_dataset_config = self.get_val_dataset_config()
        show_dataset_config = self.get_show_dataset_config()

        train_val_loader_config = self.get_train_loader_config()
        show_loader_config = self.get_show_loader_config()

        train_log_config = self.get_train_log_config()
        val_log_config = self.get_val_log_config()
        show_log_config = self.get_show_log_config()

        vsat_experiment_config = self.get_vsat_experiment_config()

        metric_config = MetricConfig().get()

        train_loop = TrainLoop(self.device, train_dataset_config, train_val_loader_config,
                               self.models, self.criterions, self.optimizers,
                               self.models_configs, self.criterions_configs, train_log_config, metric_config,
                               self.experiment_config,
                               True)

        val_loop = ValLoop(self.device, val_dataset_config, train_val_loader_config,
                           self.models, self.criterions, self.optimizers,
                           self.models_configs, self.criterions_configs, val_log_config, metric_config,
                           vsat_experiment_config, False)

        show_loop = ShowLoop(self.device, show_dataset_config, show_loader_config,
                             self.models, self.criterions, self.optimizers,
                             self.models_configs, self.criterions_configs, show_log_config, metric_config,
                             vsat_experiment_config, False)

        train_loss_event = CustomPeriodicEvent(n_iterations=train_log_config.LOSS_LOG_INTERVAL)
        train_metric_event = CustomPeriodicEvent(n_iterations=train_log_config.METRIC_LOG_INTERVAL)
        val_event = CustomPeriodicEvent(n_iterations=val_log_config.LOG_INTERVAL)
        show_event = CustomPeriodicEvent(n_epochs=show_log_config.LOG_INTERVAL)

        train_loss_event.attach(train_loop.engine)
        train_metric_event.attach(train_loop.engine)
        val_event.attach(train_loop.engine)
        show_event.attach(train_loop.engine)

        @train_loop.engine.on(train_loss_event._periodic_event_completed)
        def on_train_loss(engine):
            tb_output_losses(self.writer, engine, engine, "train")

        @train_loop.engine.on(train_metric_event._periodic_event_completed)
        def on_train_metric(engine):
            tb_output_metrics(self.writer, engine, engine, "train")

        @train_loop.engine.on(val_event._periodic_event_completed)
        def on_val(engine):
            val_loop.run()

            tb_output_losses(self.writer, val_loop.engine, engine, "val")
            tb_output_metrics(self.writer, val_loop.engine, engine, "val")

        @train_loop.engine.on(show_event._periodic_event_completed)
        def on_show(engine):
            show_loop.run()
            tb_plot_keypoints_and_descriptors(self.writer, engine.state.epoch, show_loop.engine.state.metrics[SHOW])

        return train_loop

    def bind_checkpoints(self, engine, models, optimizers):
        if self.checkpoint_dir is not None:
            checkpoint_saver = ModelCheckpoint(self.checkpoint_dir, CHECKPOINT_PREFIX,
                                               save_interval=self.experiment_config.SAVE_INTERVAL,
                                               n_saved=self.experiment_config.N_SAVED)
            engine.add_event_handler(Events.ITERATION_COMPLETED, checkpoint_saver, {MODEL: models,
                                                                                    OPTIMIZER: optimizers})

    def start_logging(self):
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def end_logging(self):
        if self.writer is not None:
            self.writer.close()


class TrainExperimentDebug(TrainExperiment):

    def get_models_configs(self):
        return DebugModelConfig().get()

    def get_experiments_configs(self):
        return DebugExperimentConfig().get()

    def get_train_dataset_config(self):
        return DebugTrainValDatasetConfig().get()

    def get_val_dataset_config(self):
        return DebugTrainValDatasetConfig().get()

    def get_show_dataset_config(self):
        return DebugShowDatasetConfig().get()

    def get_train_loader_config(self):
        return DebugLoaderConfig().get()

    def get_show_loader_config(self):
        return DebugLoaderConfig().get()

    def get_train_log_config(self):
        return DebugLogConfig().get()

    def get_val_log_config(self):
        return DebugLogConfig().get()

    def get_show_log_config(self):
        return DebugLogConfig().get()


class TestExperiment(Experiment):

    def get_models_configs(self):
        return ModelConfig().get()

    def get_criterions_configs(self):
        return CriterionConfig().get()

    def get_experiments_configs(self):
        return VSATExperimentConfig().get()

    def get_models(self, models_configs):
        model = NetVGG(models_configs.GRID_SIZE,
                       models_configs.DESCRIPTOR_SIZE,
                       models_configs.NMS_KERNEL_SIZE).to(self.device)
        return model

    def get_criterions(self, models_configs, criterions_configs):
        return None

    def get_optimizers(self, models):
        return None

    def load_checkpoints(self, models, optimizers):
        if self.checkpoint_iter is not None:
            model_path = os.path.join(self.checkpoint_dir,
                                      f"{CHECKPOINT_PREFIX}_{MODEL}_{self.checkpoint_iter}.pth")
            models.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_loop(self):
        test_dataset_config = TestDatasetConfig().get()
        test_loader_config = TestLoaderConfig().get()

        metric_config = MetricConfig().get()

        test_loop = TestLoop(self.device, test_dataset_config, test_loader_config,
                             self.models, self.criterions, self.optimizers,
                             self.models_configs, self.criterions_configs, None, metric_config,
                             self.experiment_config,
                             False)

        @test_loop.engine.on(Events.EPOCH_COMPLETED)
        def on_epoch(engine):
            rep, nn = engine.state.metrics[METRICS]
            ft, skpt, sdt = engine.state.metrics[TIMES]

            print(f"Repeatability: {rep}")
            print(f"NN Match Score: {nn}")

            print(f"Forward pass time: {ft}")
            print(f"Kp selection time: {skpt}")
            print(f"D selection time: {sdt}")

            io_plot_keypoints_and_descriptors(self.log_dir, engine.state.metrics[SHOW])

        return test_loop

    def bind_checkpoints(self, engine, models, optimizers):
        pass

    def start_logging(self):
        pass

    def end_logging(self):
        pass


class AnalyzeExperiment(Experiment):

    def get_models_configs(self):
        return ModelConfig().get()

    def get_criterions_configs(self):
        return CriterionConfig().get()

    def get_experiments_configs(self):
        return VSATExperimentConfig().get()

    def get_models(self, models_configs):
        model = NetVGG(models_configs.GRID_SIZE,
                       models_configs.DESCRIPTOR_SIZE,
                       models_configs.NMS_KERNEL_SIZE, True).to(self.device)
        return model

    def get_criterions(self, models_configs, criterions_configs):
        det_criterion = MSELoss(criterions_configs.NMS_THRESH, criterions_configs.NMS_K_SIZE,
                                criterions_configs.TOP_K,
                                criterions_configs.GAUSS_K_SIZE, criterions_configs.GAUSS_SIGMA,
                                criterions_configs.DET_LAMBDA)
        des_criterion = DenseInterQTripletLoss(models_configs.GRID_SIZE, criterions_configs.MARGIN,
                                          criterions_configs.DES_LAMBDA)
        return det_criterion, des_criterion

    def get_optimizers(self, models):
        return None

    def load_checkpoints(self, models, optimizers):
        if self.checkpoint_iter is not None:
            model_path = os.path.join(self.checkpoint_dir,
                                      f"{CHECKPOINT_PREFIX}_{MODEL}_{self.checkpoint_iter}.pth")
            models.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_loop(self):
        analyze_dataset_config = AnalyzeDatasetConfig().get()
        analyze_loader_config = AnalyzeLoaderConfig().get()

        metric_config = MetricConfig().get()

        analyze_loop = AnalyseLoop(self.device, analyze_dataset_config, analyze_loader_config,
                                   self.models, self.criterions, self.optimizers,
                                   self.models_configs, self.criterions_configs, None, metric_config,
                                   self.experiment_config,
                                   False)

        return analyze_loop

    def get_last_endpoint(self):
        return self.loop.engine.state.output

    def bind_checkpoints(self, engine, models, optimizers):
        pass

    def start_logging(self):
        pass

    def end_logging(self):
        pass