from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from Net.experiments.base.default_experiment import *
from Net.experiments.base.config import MainConfig, DebugConfig, DetectorConfig
from Net.experiments.base.metric_binder import TrainMetricBinder, TrainDetMetricBinder
from Net.source.hpatches_dataset import (
    HPatchesDataset,
    Grayscale,
    Normalize,
    RandomCrop,
    Rescale,
    ToTensor
)
from Net.source.nn.model import NetVGG
from Net.source.nn.main_criterion import MSELoss, HardQuadTripletSOSRLoss
from Net.source.utils.image_utils import select_gt_and_keypoints, warp_points, get_visible_keypoints_mask
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
        return HPatchesDataset(root_path=root_path,
                               csv_file=csv_file,
                               transform=transforms.Compose(transform),
                               include_sources=include_sources)

    def init_loaders(self):
        ds = self.config.dataset
        ls = self.config.loader

        train_transform = [Grayscale(),
                           Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                           Rescale((960, 1280)),
                           RandomCrop((720, 960)),
                           Rescale((240, 320)),
                           ToTensor()]

        val_transform = [Grayscale(),
                         Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                         Rescale((960, 1280)),
                         Rescale((240, 320)),
                         ToTensor()]

        show_transform = [Grayscale(),
                          Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                          Rescale((960, 1280)),
                          Rescale((320, 640)),
                          ToTensor()]

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
        self.criterions[DES_CRITERION] = HardQuadTripletSOSRLoss(ms.GRID_SIZE, cs.MARGIN, cs.NUM_NEG, cs.SOS_NEG,
                                                                 cs.DES_LAMBDA)

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

        des_loss1 = triplet_criterion(kp1, w_kp1, kp1_desc, desc2, homo12)
        des_loss2 = triplet_criterion(kp2, w_kp2, kp2_desc, desc1, homo21)

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

        score1, desc1 = model(image1)
        score2, desc2 = model(image2)

        _, kp1 = select_gt_and_keypoints(score1, cs.NMS_THRESH, cs.NMS_K_SIZE, cs.TOP_K)
        _, kp2 = select_gt_and_keypoints(score2, cs.NMS_THRESH, cs.NMS_K_SIZE, cs.TOP_K)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        kp1_desc = sample_descriptors(desc1, kp1, ms.GRID_SIZE)
        kp2_desc = sample_descriptors(desc2, kp2, ms.GRID_SIZE)

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
        }

    def bind_events(self):
        TrainMetricBinder(self.config, self.writer).bind(self.engines, self.loaders)

    def analyze_inference(self):
        self.init_config()

        ds = self.config.dataset
        ls = self.config.loader

        analyze_transform = [Grayscale(),
                             Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                             Rescale((960, 1280)),
                             Rescale((320, 640)),
                             ToTensor()]

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

            _, kp1 = select_gt_and_keypoints(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
            _, kp2 = select_gt_and_keypoints(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

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


class TED(TE):

    def init_config(self):
        self.config = DetectorConfig()

    def init_models(self):
        ms = self.config.model
        self.models[MODEL] = NetVGG(ms.GRID_SIZE, ms.DESCRIPTOR_SIZE, ms.NMS_KERNEL_SIZE, True).to(self.device)

    def iteration(self, engine, batch):
        model = self.models[MODEL]

        mse_criterion = self.criterions[DET_CRITERION]

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, _, _ = model(image1)
        score2, _, _ = model(image2)

        det_loss1, kp1 = mse_criterion(score1, score2, homo12)
        det_loss2, kp2 = mse_criterion(score2, score1, homo21)

        w_kp1 = warp_points(kp1, homo12)
        w_kp2 = warp_points(kp2, homo21)

        wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
        wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

        det_loss = (det_loss1 + det_loss2) / 2

        return {
            LOSS: det_loss,

            KP1: kp1,
            KP2: kp2,

            W_KP1: w_kp1,
            W_KP2: w_kp2,

            WV_KP2_MASK: wv_kp2_mask,
            WV_KP1_MASK: wv_kp1_mask
        }

    def inference(self, engine, batch):
        model = self.models[MODEL]

        ls = self.config.criterion

        image1, image2, homo12, homo21 = (
            batch[IMAGE1].to(self.device),
            batch[IMAGE2].to(self.device),
            batch[HOMO12].to(self.device),
            batch[HOMO21].to(self.device)
        )

        score1, _, _ = model(image1)
        score2, _, _ = model(image2)

        _, kp1 = select_gt_and_keypoints(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
        _, kp2 = select_gt_and_keypoints(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

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
            WV_KP1_MASK: wv_kp1_mask
        }

    def bind_events(self):
        TrainDetMetricBinder(self.config, self.writer).bind(self.engines, self.loaders)

    def analyze_inference(self):
        self.init_config()

        ds = self.config.dataset
        ls = self.config.loader

        analyze_transform = [Grayscale(),
                             Normalize(mean=ds.DATASET_MEAN, std=ds.DATASET_STD),
                             Rescale((960, 1280)),
                             Rescale((320, 640)),
                             ToTensor()]

        analyze_loader = DataLoader(
            TE.get_dataset(os.path.join("../", ds.DATASET_ROOT),
                           ds.ANALYZE_CSV, analyze_transform, True),
            batch_size=ls.ANALYZE_BATCH_SIZE)

        self.init_models()

        self.load_checkpoint()

        self.models[MODEL].eval()

        with torch.no_grad():
            batch = analyze_loader.__iter__().__next__()
            model = self.models[MODEL]

            ls = self.config.criterion

            image1, image2, homo12, homo21 = (
                batch[IMAGE1].to(self.device),
                batch[IMAGE2].to(self.device),
                batch[HOMO12].to(self.device),
                batch[HOMO21].to(self.device)
            )

            score1, _, debug1 = model(image1)
            score2, _, debug2 = model(image2)

            _, kp1 = select_gt_and_keypoints(score1, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)
            _, kp2 = select_gt_and_keypoints(score2, ls.NMS_THRESH, ls.NMS_K_SIZE, ls.TOP_K)

            w_kp1 = warp_points(kp1, homo12)
            w_kp2 = warp_points(kp2, homo21)

            wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)
            wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)

            output = {
                S_IMAGE1: batch[S_IMAGE1],
                S_IMAGE2: batch[S_IMAGE2],

                DEBUG1: debug1,
                DEBUG2: debug2,

                HOMO12: homo12,
                HOMO21: homo21,

                SCORE1: score1,
                SCORE2: score2,

                KP1: kp1,
                KP2: kp2,

                W_KP1: w_kp1,
                W_KP2: w_kp2,

                WV_KP2_MASK: wv_kp2_mask,
                WV_KP1_MASK: wv_kp1_mask
            }

        return output
