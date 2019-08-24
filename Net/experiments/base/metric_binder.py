from ignite.contrib.handlers.custom_events import CustomPeriodicEvent

from Net.experiments.base.base_experiment import BaseMetricBinder
from Net.experiments.base.default_experiment import *

from Net.source.utils.ignite_utils import PeriodicMetric, AveragePeriodicMetric, AveragePeriodicListMetric, \
    CollectMetric
from Net.source.utils.eval_utils import repeatability_score, metric_scores, plot_keypoints_and_descriptors

"""
Metric keys
"""
METRICS = 'main_metrics'
REP_SCORE = 'repeatability_score'
NN_MATCH_SCORE = 'nearest_neighbour_match_score'
NNT_MATCH_SCORE = 'nearest_neighbour_thresh_match_score'
NNR_MATCH_SCORE = 'nearest_neighbour_ratio_match_score'

SHOW = 'show'


def output_losses(writer, data_engine, state_engine, tag):
    writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{DET_LOSS}", data_engine.state.metrics[DET_LOSS], state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{DES_LOSS}", data_engine.state.metrics[DES_LOSS], state_engine.state.iteration)


def output_loss(writer, data_engine, state_engine, tag):
    writer.add_scalar(f"{tag}/{LOSS}", data_engine.state.metrics[LOSS], state_engine.state.iteration)


def output_metrics(writer, data_engine, state_engine, tag):
    rep, nn, nnt, nnr = data_engine.state.metrics[METRICS]
    writer.add_scalar(f"{tag}/{REP_SCORE}", rep, state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{NN_MATCH_SCORE}", nn, state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{NNT_MATCH_SCORE}", nnt, state_engine.state.iteration)
    writer.add_scalar(f"{tag}/{NNR_MATCH_SCORE}", nnr, state_engine.state.iteration)


def output_metric(writer, data_engine, state_engine, tag):
    writer.add_scalar(f"{tag}/{REP_SCORE}", data_engine.state.metrics[REP_SCORE], state_engine.state.iteration)


class TrainMetricBinder(BaseMetricBinder):

    def bind(self, engines, loaders):
        train_engine = engines[TRAIN_ENGINE]
        val_engine = engines[VAL_ENGINE]
        show_engine = engines[SHOW_ENGINE]

        ls = self.config.log
        ms = self.config.metric

        def l_loss(x):
            return x[LOSS]

        def l_det_loss(x):
            return x[DET_LOSS]

        def l_des_loss(x):
            return x[DES_LOSS]

        def l_metrics(x):
            ms1 = metric_scores(x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], x[KP1_DESC],
                                                                  x[KP2_DESC], ms.DET_THRESH, ms.DES_THRESH, ms.DES_RATIO)
            ms2 = metric_scores(x[KP2], x[W_KP1], x[KP1], x[WV_KP1_MASK], x[KP2_DESC],
                                                                  x[KP1_DESC], ms.DET_THRESH, ms.DES_THRESH, ms.DES_RATIO)
            return [(m1 + m2) / 2 for m1, m2 in zip(ms1, ms2)]

        def l_collect_show(x):
            return x[S_IMAGE1][0], x[S_IMAGE2][0], x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], x[KP1_DESC], x[KP2_DESC]

        # Attach train metrics
        AveragePeriodicMetric(l_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, LOSS)
        AveragePeriodicMetric(l_det_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, DET_LOSS)
        AveragePeriodicMetric(l_des_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, DES_LOSS)
        PeriodicMetric(l_metrics, ls.TRAIN.METRIC_LOG_INTERVAL).attach(train_engine, METRICS)

        # Val metrics
        AveragePeriodicMetric(l_loss).attach(val_engine, LOSS)
        AveragePeriodicMetric(l_det_loss).attach(val_engine, DET_LOSS)
        AveragePeriodicMetric(l_des_loss).attach(val_engine, DES_LOSS)
        AveragePeriodicListMetric(l_metrics).attach(val_engine, METRICS)

        # Show metrics
        CollectMetric(l_collect_show).attach(show_engine, SHOW)

        train_loss_event = CustomPeriodicEvent(n_iterations=ls.TRAIN.LOSS_LOG_INTERVAL)
        train_metric_event = CustomPeriodicEvent(n_iterations=ls.TRAIN.METRIC_LOG_INTERVAL)

        val_event = CustomPeriodicEvent(n_iterations=ls.VAL.LOG_INTERVAL)
        show_event = CustomPeriodicEvent(n_epochs=ls.SHOW.LOG_INTERVAL)

        val_loader = loaders[VAL_LOADER]
        show_loader = loaders[SHOW_LOADER]

        # Attach events to train engine
        train_loss_event.attach(train_engine)
        train_metric_event.attach(train_engine)

        val_event.attach(train_engine)
        show_event.attach(train_engine)

        # Train events
        @train_engine.on(train_loss_event._periodic_event_completed)
        def on_tle(engine):
            output_losses(self.writer, engine, engine, "train")

        @train_engine.on(train_metric_event._periodic_event_completed)
        def on_tme(engine):
            output_metrics(self.writer, engine, engine, "train")

        # Val event
        @train_engine.on(val_event._periodic_event_completed)
        def on_ve(engine):
            val_engine.run(val_loader)

            output_losses(self.writer, val_engine, train_engine, "val")
            output_metrics(self.writer, val_engine, train_engine, "val")

        # Show event
        @train_engine.on(show_event._periodic_event_completed)
        def on_se(engine):
            show_engine.run(show_loader)

            plot_keypoints_and_descriptors(self.writer, train_engine.state.epoch, show_engine.state.metrics[SHOW],
                                           ms.DET_THRESH)


class TrainDetMetricBinder(BaseMetricBinder):

    def bind(self, engines, loaders):
        train_engine = engines[TRAIN_ENGINE]
        val_engine = engines[VAL_ENGINE]
        show_engine = engines[SHOW_ENGINE]

        ls = self.config.log
        ms = self.config.metric

        def l_loss(x):
            return x[LOSS]

        def l_rep_score(x):
            rep1 = repeatability_score(x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], ms.DET_THRESH)
            rep2 = repeatability_score(x[KP2], x[W_KP1], x[KP1], x[WV_KP1_MASK], ms.DET_THRESH)
            return (rep1 + rep2) / 2

        def l_collect_show(x):
            return x[S_IMAGE1][0], x[S_IMAGE2][0], x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], None, None

        # Train metrics
        AveragePeriodicMetric(l_loss, ls.TRAIN.LOSS_LOG_INTERVAL).attach(train_engine, LOSS)
        PeriodicMetric(l_rep_score, ls.TRAIN.MAIN_METRIC_LOG_INTERVAL).attach(train_engine, REP_SCORE)

        # Val metrics
        AveragePeriodicMetric(l_loss).attach(val_engine, LOSS)
        AveragePeriodicMetric(l_rep_score).attach(val_engine, REP_SCORE)

        # Show metrics
        CollectMetric(l_collect_show).attach(show_engine, SHOW)

        train_loss_event = CustomPeriodicEvent(n_iterations=ls.TRAIN.LOSS_LOG_INTERVAL)
        train_metric_event = CustomPeriodicEvent(n_iterations=ls.TRAIN.METRIC_LOG_INTERVAL)

        val_event = CustomPeriodicEvent(n_iterations=ls.VAL.LOG_INTERVAL)
        show_event = CustomPeriodicEvent(n_epochs=ls.SHOW.LOG_INTERVAL)

        val_loader = loaders[VAL_LOADER]
        show_loader = loaders[SHOW_LOADER]

        # Attach events to train engine
        train_loss_event.attach(train_engine)
        train_metric_event.attach(train_engine)

        val_event.attach(train_engine)
        show_event.attach(train_engine)

        # Train events
        @train_engine.on(train_loss_event._periodic_event_completed)
        def on_tle(engine):
            output_loss(self.writer, train_engine, train_engine, "train")

        @train_engine.on(train_metric_event._periodic_event_completed)
        def on_tme(engine):
            output_metric(self.writer, train_engine, train_engine, "train")

        # Val event
        @train_engine.on(val_event._periodic_event_completed)
        def on_ve(engine):
            val_engine.run(val_loader)

            output_loss(self.writer, val_engine, train_engine, "val")
            output_metric(self.writer, val_engine, train_engine, "val")

        # Show event
        @train_engine.on(show_event._periodic_event_completed)
        def on_se(engine):
            show_engine.run(show_loader)

            plot_keypoints_and_descriptors(self.writer, train_engine.state.epoch, show_engine.state.metrics[SHOW],
                                           ms.DET_THRESH)
