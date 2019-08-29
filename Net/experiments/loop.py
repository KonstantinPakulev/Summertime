import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from Net.experiments.base import Loop

from Net.source.hpatches_dataset import (
    HPatchesDataset,
    PhotometricAugmentation,
    ResizeItem,
    CropItem,
    FinishAugmentation,

    IMAGE1,
    IMAGE2,
    HOMO12,
    HOMO21,
    S_IMAGE1,
    S_IMAGE2
)

from Net.source.utils.model_utils import sample_descriptors
from Net.source.utils.image_utils import warp_image, select_keypoints, warp_points, get_visible_keypoints_mask
from Net.source.utils.eval_utils import metric_scores
from Net.source.utils.ignite_utils import PeriodicMetric, AveragePeriodicMetric, AveragePeriodicListMetric, \
    CollectMetric

"""
Endpoint keys
"""
SCORE1 = 'score1'
SCORE2 = 'score2'
KP1 = 'kp1'
KP2 = 'kp2'
KP1_DESC = 'kp1_desc'
KP2_DESC = 'kp2_desc'
DESC1 = 'desc1'
DESC2 = 'desc2'
W_VIS_MASK1 = 'w_vis_mask1'
W_VIS_MASK2 = 'w_vis_mask2'
W_KP1 = 'w_kp1'
W_KP2 = 'w_kp2'
WV_KP1_MASK = 'wv_kp1_mask'
WV_KP2_MASK = 'wv_kp2_mask'

DET_THRESH = 'det_thresh'

LOSS = 'loss'
DET_LOSS = 'det_loss'
DES_LOSS = 'des_loss'

METRICS = 'metrics'
SHOW = 'show'
TIMES = 'times'

FORWARD_TIME1 = 'forward_time1'
FORWARD_TIME2 = 'forward_time2'
SELECT_KP1_TIME = 'select_kp1_time'
SELECT_KP2_TIME = 'select_kp2_time'
SELECT_DESC1_TIME = 'select_desc1_time'
SELECT_DESC2_TIME = 'select_desc2_time'

DEBUG1 = 'debug1'
DEBUG2 = 'debug2'

"""
Metric collectors
"""


def l_loss(x):
    return x[LOSS]


def l_det_loss(x):
    return x[DET_LOSS]


def l_des_loss(x):
    return x[DES_LOSS]


def l_metrics(x):
    ms1 = metric_scores(x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], x[KP1_DESC],
                        x[KP2_DESC], x[DET_THRESH])
    ms2 = metric_scores(x[KP2], x[W_KP1], x[KP1], x[WV_KP1_MASK], x[KP2_DESC],
                        x[KP1_DESC], x[DET_THRESH])
    return [(m1 + m2) / 2 for m1, m2 in zip(ms1, ms2)]


def l_collect_show(x):
    return x[S_IMAGE1][0], x[S_IMAGE2][0], x[KP1], x[W_KP2], x[KP2], x[WV_KP2_MASK], x[KP1_DESC], x[KP2_DESC], x[
        DET_THRESH]


def l_times(x):
    ts1 = [x[FORWARD_TIME1], x[SELECT_KP1_TIME], x[SELECT_DESC1_TIME]]
    ts2 = [x[FORWARD_TIME2], x[SELECT_KP2_TIME], x[SELECT_DESC2_TIME]]
    return [(t1 + t2) / 2 for t1, t2 in zip(ts1, ts2)]


"""
Loops
"""


class TrainLoop(Loop):

    def get_dataset(self, dataset_config):
        train_transform = [PhotometricAugmentation(brightness=None, contrast=None),
                           ResizeItem((960, 1280)),
                           CropItem((720, 960)),
                           ResizeItem((dataset_config.HEIGHT, dataset_config.WIDTH)),
                           FinishAugmentation(None, None)]

        dataset = HPatchesDataset(dataset_config.DATASET_ROOT,
                                  dataset_config.CSV,
                                  item_transforms=transforms.Compose(train_transform),
                                  include_sources=dataset_config.INCLUDE_SOURCES)
        return dataset

    def get_loader(self, dataset, loader_config):
        loader = DataLoader(dataset, batch_size=loader_config.BATCH_SIZE, shuffle=loader_config.SHUFFLE,
                            num_workers=loader_config.NUM_WORKERS)
        return loader

    def forward(self, engine, batch):
        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        model = self.models

        score1, desc1 = model(image1)
        score2, desc2 = model(image2)

        kp1 = select_keypoints(score1, self.criterions_configs.NMS_THRESH, self.criterions_configs.NMS_K_SIZE,
                               self.criterions_configs.TOP_K)
        kp2 = select_keypoints(score2, self.criterions_configs.NMS_THRESH, self.criterions_configs.NMS_K_SIZE,
                               self.criterions_configs.TOP_K)

        kp1_desc = sample_descriptors(desc1, kp1, self.models_configs.GRID_SIZE)
        kp2_desc = sample_descriptors(desc2, kp2, self.models_configs.GRID_SIZE)

        w_vis_mask1 = warp_image(score2, torch.ones_like(score1).to(score1.device), homo21).gt(0)
        w_vis_mask2 = warp_image(score1, torch.ones_like(score2).to(score2.device), homo12).gt(0)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)
        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)

        return {
            SCORE1: score1,
            SCORE2: score2,

            KP1: kp1,
            KP2: kp2,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc,

            DESC1: desc1,
            DESC2: desc2,

            W_VIS_MASK1: w_vis_mask1,
            W_VIS_MASK2: w_vis_mask2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP1_MASK: wv_kp1_mask,
            WV_KP2_MASK: wv_kp2_mask,

            HOMO12: homo12,
            HOMO21: homo21
        }

    def calculate_losses(self, engine, endpoint):
        det_criterion, des_criterion = self.criterions

        score1, score2, homo12, homo21, w_vis_mask1, w_vis_mask2, desc1, desc2 = (
            endpoint[SCORE1],
            endpoint[SCORE2],
            endpoint[HOMO12],
            endpoint[HOMO21],
            endpoint[W_VIS_MASK1],
            endpoint[W_VIS_MASK2],
            endpoint[DESC1],
            endpoint[DESC2]
        )

        det_loss1 = det_criterion(score1, score2, w_vis_mask2, homo12)
        det_loss2 = det_criterion(score2, score1, w_vis_mask1, homo21)

        det_loss = (det_loss1 + det_loss2) / 2

        des_loss1 = des_criterion(desc1, desc2, homo12, w_vis_mask1, score2)
        des_loss2 = des_criterion(desc2, desc1, homo21, w_vis_mask2, score1)

        des_loss = (des_loss1 + des_loss2) / 2

        loss = det_loss + des_loss

        endpoint[LOSS] = loss
        endpoint[DET_LOSS] = det_loss
        endpoint[DES_LOSS] = des_loss

    def step(self, engine, endpoint):
        loss = endpoint[LOSS]

        optimizer = self.optimizers

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def finalize_endpoint(self, engine, batch, endpoint):
        endpoint[DET_THRESH] = self.metric_config.DET_THRESH

    def bind_metrics(self, engine):
        AveragePeriodicMetric(l_loss, self.log_config.LOSS_LOG_INTERVAL).attach(engine, LOSS)
        AveragePeriodicMetric(l_det_loss, self.log_config.LOSS_LOG_INTERVAL).attach(engine, DET_LOSS)
        AveragePeriodicMetric(l_des_loss, self.log_config.LOSS_LOG_INTERVAL).attach(engine, DES_LOSS)
        PeriodicMetric(l_metrics, self.log_config.METRIC_LOG_INTERVAL).attach(engine, METRICS)


class ValLoop(TrainLoop):

    def get_dataset(self, dataset_config):
        val_transform = [PhotometricAugmentation(brightness=None, contrast=None),
                         ResizeItem((960, 1280)),
                         ResizeItem((dataset_config.HEIGHT, dataset_config.WIDTH)),
                         FinishAugmentation(None, None)]

        dataset = HPatchesDataset(dataset_config.DATASET_ROOT,
                                  dataset_config.CSV,
                                  item_transforms=transforms.Compose(val_transform),
                                  include_sources=dataset_config.INCLUDE_SOURCES)
        return dataset

    def finalize_endpoint(self, engine, batch, endpoint):
        super().finalize_endpoint(engine, batch, endpoint)

        if S_IMAGE1 in batch:
            endpoint[S_IMAGE1] = batch[S_IMAGE1]
        if S_IMAGE2 in batch:
            endpoint[S_IMAGE2] = batch[S_IMAGE2]

    def step(self, engine, endpoint):
        pass

    def bind_metrics(self, engine):
        AveragePeriodicMetric(l_loss).attach(engine, LOSS)
        AveragePeriodicMetric(l_det_loss).attach(engine, DET_LOSS)
        AveragePeriodicMetric(l_des_loss).attach(engine, DES_LOSS)
        AveragePeriodicListMetric(l_metrics).attach(engine, METRICS)


class TestLoop(ValLoop):

    def forward(self, engine, batch):
        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        model = self.models

        sf1, ef1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        sf2, ef2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        sskp1, eskp1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        sskp2, eskp2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        ssd1, esd1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ssd2, esd2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        sf1.record()
        score1, desc1 = model(image1)
        ef1.record()

        sf2.record()
        score2, desc2 = model(image2)
        ef2.record()

        sskp1.record()
        kp1 = select_keypoints(score1, self.criterions_configs.NMS_THRESH, self.criterions_configs.NMS_K_SIZE,
                               self.criterions_configs.TOP_K)
        eskp1.record()

        sskp2.record()
        kp2 = select_keypoints(score2, self.criterions_configs.NMS_THRESH, self.criterions_configs.NMS_K_SIZE,
                               self.criterions_configs.TOP_K)
        eskp2.record()

        ssd1.record()
        kp1_desc = sample_descriptors(desc1, kp1, self.models_configs.GRID_SIZE)
        esd1.record()

        ssd2.record()
        kp2_desc = sample_descriptors(desc2, kp2, self.models_configs.GRID_SIZE)
        esd2.record()

        torch.cuda.synchronize()

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)
        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)

        f1 = sf1.elapsed_time(ef1)
        f2 = sf2.elapsed_time(ef2)

        skp1 = sskp1.elapsed_time(eskp1)
        skp2 = sskp2.elapsed_time(eskp2)

        sd1 = ssd1.elapsed_time(esd1)
        sd2 = ssd2.elapsed_time(esd2)

        return {
            KP1: kp1,
            KP2: kp2,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP1_MASK: wv_kp1_mask,
            WV_KP2_MASK: wv_kp2_mask,

            FORWARD_TIME1: f1,
            FORWARD_TIME2: f2,

            SELECT_KP1_TIME: skp1,
            SELECT_KP2_TIME: skp2,

            SELECT_DESC1_TIME: sd1,
            SELECT_DESC2_TIME: sd2
        }

    def calculate_losses(self, engine, endpoint):
        pass

    def bind_metrics(self, engine):
        AveragePeriodicListMetric(l_metrics).attach(engine, METRICS)
        AveragePeriodicListMetric(l_times).attach(engine, TIMES)
        CollectMetric(l_collect_show).attach(engine, SHOW)


class ShowLoop(ValLoop):

    def calculate_losses(self, engine, endpoint):
        pass

    def bind_metrics(self, engine):
        CollectMetric(l_collect_show).attach(engine, SHOW)


class AnalyseLoop(ValLoop):

    def forward(self, engine, batch):
        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        model = self.models

        score1, desc1, debug1 = model(image1)
        score2, desc2, debug2 = model(image2)

        kp1 = select_keypoints(score1, self.criterions_configs.NMS_THRESH, self.criterions_configs.NMS_K_SIZE,
                               self.criterions_configs.TOP_K)
        kp2 = select_keypoints(score2, self.criterions_configs.NMS_THRESH, self.criterions_configs.NMS_K_SIZE,
                               self.criterions_configs.TOP_K)

        kp1_desc = sample_descriptors(desc1, kp1, self.models_configs.GRID_SIZE)
        kp2_desc = sample_descriptors(desc2, kp2, self.models_configs.GRID_SIZE)

        w_vis_mask1 = warp_image(score2, torch.ones_like(score1).to(score1.device), homo21).gt(0)
        w_vis_mask2 = warp_image(score1, torch.ones_like(score2).to(score2.device), homo12).gt(0)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)
        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)

        return {
            SCORE1: score1,
            SCORE2: score2,

            KP1: kp1,
            KP2: kp2,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc,

            DESC1: desc1,
            DESC2: desc2,

            W_VIS_MASK1: w_vis_mask1,
            W_VIS_MASK2: w_vis_mask2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP1_MASK: wv_kp1_mask,
            WV_KP2_MASK: wv_kp2_mask,

            HOMO12: homo12,
            HOMO21: homo21,

            DEBUG1: debug1,
            DEBUG2: debug2
        }
