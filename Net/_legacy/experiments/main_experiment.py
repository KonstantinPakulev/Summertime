from Net.experiments.loop import *
from Net.experiments.config import MainConfig
from Net._legacy.experiments.metric_binder import TrainMetricBinder, TestMetricBinder
from Net._legacy.source.hpatches_dataset_old import (
    HPatchesDatasetOld,
    GrayscaleOld,
    NormalizeOld,
    RandomCropOld,
    RescaleOld,
    ToTensorOld
)
from Net.source.nn.model import NetVGG
from Net.source.nn.criterion import MSELoss, DenseQTripletLoss
from Net.source.utils.image_utils import prepare_gt_score, warp_points, get_visible_keypoints_mask
from Net.source.utils.model_utils import sample_descriptors


# noinspection PyMethodMayBeStatic
class TE(DefaultExperiment):
    """
    Current main train experiment
    """

    def init_config(self):
        self.config = MainConfig()

    @staticmethod
    def get_dataset(root_path, csv_file, transform, include_sources):
        return HPatchesDatasetOld(root_path=root_path,
                                  csv_file=csv_file,
                                  transform=transforms.Compose(transform),
                                  include_sources=include_sources)

    def init_loaders(self):
        ds = self.config.dataset
        ls = self.config.loader

        train_transform = [GrayscaleOld(),
                           NormalizeOld(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                           RescaleOld((960, 1280)),
                           RandomCropOld((720, 960)),
                           RescaleOld((240, 320)),
                           ToTensorOld()]

        val_transform = [GrayscaleOld(),
                         NormalizeOld(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                         RescaleOld((960, 1280)),
                         RescaleOld((240, 320)),
                         ToTensorOld()]

        show_transform = [GrayscaleOld(),
                          NormalizeOld(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                          RescaleOld((960, 1280)),
                          RescaleOld((320, 640)),
                          ToTensorOld()]

        self.loaders[TRAIN_LOADER] = DataLoader(
            TE.get_dataset(ds.DATASET_ROOT, ds.TRAIN_CSV, train_transform, False),
            batch_size=ls.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=ls.NUM_WORKERS)
        self.loaders[VAL_LOADER] = DataLoader(
            TE.get_dataset(ds.DATASET_ROOT, ds.VAL_CSV, val_transform, False),
            batch_size=ls.VAL_BATCH_SIZE,
            shuffle=True,
            num_workers=ls.NUM_WORKERS)
        self.loaders[SHOW_LOADER] = DataLoader(
            TE.get_dataset(ds.DATASET_ROOT, ds.SHOW_CSV, show_transform, True),
            batch_size=ls.SHOW_BATCH_SIZE)

    def init_models(self):
        ms = self.config.model
        self.models[MODEL] = NetVGG(ms.GRID_SIZE, ms.DESCRIPTOR_SIZE, ms.NMS_KERNEL_SIZE).to(self.device)

    def init_criterions(self):
        ms = self.config.model
        cs = self.config.criterion

        self.criterions[DET_CRITERION] = MSELoss(cs.NMS_THRESH, cs.NMS_K_SIZE,
                                                 cs.TOP_K,
                                                 cs.GAUSS_K_SIZE, cs.GAUSS_SIGMA, cs.DET_LAMBDA)
        self.criterions[DES_CRITERION] = DenseQTripletLoss(ms.GRID_SIZE, cs.MARGIN)

    def init_optimizers(self):
        self.optimizers[OPTIMIZER] = Adam(self.models[MODEL].parameters())

    def iteration(self, engine, batch):
        model = self.models[MODEL]

        mse_criterion = self.criterions[DET_CRITERION]
        triplet_criterion = self.criterions[DES_CRITERION]

        ms = self.config.model

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, desc1 = model(image1)
        score2, desc2 = model(image2)

        det_loss1, kp1 = mse_criterion(score1, score2, homo12)
        det_loss2, kp2 = mse_criterion(score2, score1, homo21)

        kp1_desc = sample_descriptors(desc1, kp1, ms.GRID_SIZE)
        kp2_desc = sample_descriptors(desc2, kp2, ms.GRID_SIZE)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        des_loss1 = triplet_criterion(score1, score2, desc1, desc2, homo12, homo21)
        des_loss2 = triplet_criterion(score2, score1, desc2, desc1, homo21, homo12)

        det_loss = (det_loss1 + det_loss2) / 2
        des_loss = (des_loss1 + des_loss2) / 2

        loss = det_loss + des_loss

        return {
            LOSS: loss,
            DET_LOSS: det_loss,
            DES_LOSS: des_loss,

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP2_MASK: wv_kp2_mask,
            WV_KP1_MASK: wv_kp1_mask,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc
        }

    def inference(self, engine, batch):
        model = self.models[MODEL]

        ms = self.config.model
        cs = self.config.criterion

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        sf1, ef1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        sf2, ef2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        sskp1, eskp1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        sskp2, eskp2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        ssd1, esd1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ssd2, esd2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        sf1.record()
        score1, desc1 = model(image1)
        ef1.record()

        sskp1.record()
        _, kp1 = prepare_gt_score(score1, cs.NMS_THRESH, cs.NMS_K_SIZE, cs.TOP_K)
        eskp1.record()

        ssd1.record()
        kp1_desc = sample_descriptors(desc1, kp1, ms.GRID_SIZE)
        esd1.record()

        sf2.record()
        score2, desc2 = model(image2)
        ef2.record()

        sskp2.record()
        _, kp2 = prepare_gt_score(score2, cs.NMS_THRESH, cs.NMS_K_SIZE, cs.TOP_K)
        eskp2.record()

        ssd2.record()
        kp2_desc = sample_descriptors(desc2, kp2, ms.GRID_SIZE)
        esd2.record()

        torch.cuda.synchronize()

        f1 = sf1.elapsed_time(ef1)
        f2 = sf2.elapsed_time(ef2)

        skp1 = sskp1.elapsed_time(eskp1)
        skp2 = sskp2.elapsed_time(eskp2)

        sd1 = ssd1.elapsed_time(esd1)
        sd2 = ssd2.elapsed_time(esd2)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        return {
            S_IMAGE1: batch[S_IMAGE1],
            S_IMAGE2: batch[S_IMAGE2],

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP2_MASK: wv_kp2_mask,
            WV_KP1_MASK: wv_kp1_mask,

            KP1_DESC: kp1_desc,
            KP2_DESC: kp2_desc,

            FORWARD_TIME1: f1,
            FORWARD_TIME2: f2,
            SELECT_KP1_TIME: skp1,
            SELECT_KP2_TIME: skp2,
            SELECT_DESC1_TIME: sd1,
            SELECT_DESC2_TIME: sd2
        }

    def bind_events(self):
        TrainMetricBinder(self.config, self.writer).bind(self.engines, self.loaders)

    def analyze_inference(self):
        self.init_config()

        ds = self.config.dataset
        ls = self.config.loader

        analyze_transform = [GrayscaleOld(),
                             NormalizeOld(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                             RescaleOld((960, 1280)),
                             RescaleOld((320, 640)),
                             ToTensorOld()]

        analyze_loader = DataLoader(
            TE.get_dataset(os.path.join("../", ds.DATASET_ROOT),
                           ds.ANALYZE_CSV, analyze_transform, True),
            batch_size=ls.ANALYZE_BATCH_SIZE)

        ms = self.config.model
        self.models[MODEL] = NetVGG(ms.GRID_SIZE, ms.DESCRIPTOR_SIZE, ms.NMS_KERNEL_SIZE, True).to(self.device)

        self.load_checkpoint()

        self.models[MODEL].eval()

        with torch.no_grad():
            batch = analyze_loader.__iter__().__next__()
            model = self.models[MODEL]

            ms = self.config.model
            ls = self.config.criterion

            image1, image2, homo12, homo21 = (
                batch[IMAGE1].to(self.device),
                batch[IMAGE2].to(self.device),
                batch[HOMO12].to(self.device),
                batch[HOMO21].to(self.device)
            )

            score1, desc1, debug1 = model(image1)
            score2, desc2, debug2 = model(image2)

            _, kp1 = prepare_gt_score(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
            _, kp2 = prepare_gt_score(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

            kp1_desc = sample_descriptors(desc1, kp1, ms.GRID_SIZE)
            kp2_desc = sample_descriptors(desc2, kp2, ms.GRID_SIZE)

            w_kp1 = warp_points(kp1, homo12)
            w_kp2 = warp_points(kp2, homo21)

            wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
            wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

            output = {
                S_IMAGE1: batch[S_IMAGE1],
                S_IMAGE2: batch[S_IMAGE2],

                HOMO12: homo12,
                HOMO21: homo21,

                SCORE1: score1,
                SCORE2: score2,

                DESC1: desc1,
                DESC2: desc2,

                KP1: kp1,
                KP2: kp2,

                W_KP1: w_kp1,
                W_KP2: w_kp2,

                WV_KP2_MASK: wv_kp2_mask,
                WV_KP1_MASK: wv_kp1_mask,

                KP1_DESC: kp1_desc,
                KP2_DESC: kp2_desc,

                DEBUG1: debug1,
                DEBUG2: debug2
            }

        return output


    def run_test(self):
        self.init_config()

        ds = self.config.dataset
        ls = self.config.loader

        test_transform = [GrayscaleOld(),
                          NormalizeOld(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                          RescaleOld((960, 1280)),
                          RescaleOld((240, 320)),
                          ToTensorOld()]

        test_loader = DataLoader(
            TE.get_dataset(ds.DATASET_ROOT,
                           ds.TEST_CSV, test_transform, True),
            batch_size=ls.TEST_BATCH_SIZE)

        ms = self.config.model
        self.models[MODEL] = NetVGG(ms.GRID_SIZE, ms.DESCRIPTOR_SIZE, ms.NMS_KERNEL_SIZE).to(self.device)

        self.load_checkpoint()

        def test_iteration(engine, batch):
            self.models[MODEL].eval()

            with torch.no_grad():
                endpoint = self.inference(engine, batch)

            return endpoint

        engine = Engine(test_iteration)

        TestMetricBinder(self.config, None).bind(engine, test_loader)

        engine.run(test_loader)


class TrainSOSRExperiment(TrainExperiment):

    def get_criterions(self, models_configs, criterions_configs):
        det_criterion = MSELoss(criterions_configs.NMS_THRESH, criterions_configs.NMS_K_SIZE,
                                criterions_configs.TOP_K,
                                criterions_configs.GAUSS_K_SIZE, criterions_configs.GAUSS_SIGMA,
                                criterions_configs.DET_LAMBDA)
        des_criterion = DenseInterQTripletSOSRLoss(models_configs.GRID_SIZE, criterions_configs.MARGIN,
                                                   criterions_configs.DES_LAMBDA)
        return det_criterion, des_criterion
