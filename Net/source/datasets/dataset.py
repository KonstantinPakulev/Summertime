from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader

import Net.source.core.loop as l
import Net.source.core.experiment as exp
import Net.source.datasets.dataset_utils as du
import Net.source.datasets.hpatches.hpatches_transforms as ht
import Net.source.datasets.aachen.aachen_transforms as at
import Net.source.datasets.megadepth.megadepth_transforms as mt

from Net.source.datasets.hpatches.hpatches_dataset import HPatchesDataset
from Net.source.datasets.aachen.aachen_dataset import AachenDataset
from Net.source.datasets.megadepth.megadepth_dataset import MegaDepthDataset


def create_dataset(dataset_config, model_config, mode):
    datasets = []

    for d_key, d_config in dataset_config.items():

        if d_key in [du.HPATCHES_VIEW, du.HPATCHES_ILLUM]:
            size = (d_config[du.HEIGHT], d_config[du.WIDTH])

            if mode in [l.TEST, l.ANALYZE]:
                item_transforms = [ht.HPatchesToPILImage()]

                if model_config[exp.INPUT_CHANNELS] == 1:
                    item_transforms += [ht.HPatchesGrayScale()]

                item_transforms += [ht.HPatchesResize(du.AspectResize(size, False)),
                                    ht.HPatchesCrop(du.CentralCrop(size, False)),
                                    ht.HPatchesToTensor()]

                dataset = HPatchesDataset.from_config(d_config, item_transforms)

            else:
                raise NotImplementedError

        elif d_key == du.MEGADEPTH:
            size = (d_config[du.HEIGHT], d_config[du.WIDTH])

            if mode in l.TRAIN:
                item_transforms = [mt.MegaDepthToPILImage()]

                if model_config[exp.INPUT_CHANNELS] == 1:
                    item_transforms += [mt.MegaDepthToGrayScale()]

                item_transforms += [mt.MegaDepthCrop(mt.MegaDepthSharedAreaCrop()),
                                    mt.MegaDepthResize(du.AspectResize(size, True)),
                                    mt.MegaDepthCrop(du.RandomCrop(size)),
                                    mt.MegaDepthToTensor()]

            elif mode in [l.VAL, l.VISUALIZE]:
                item_transforms = [mt.MegaDepthToPILImage()]

                if model_config[exp.INPUT_CHANNELS] == 1:
                    item_transforms += [mt.MegaDepthToGrayScale()]

                item_transforms += [mt.MegaDepthResize(du.AspectResize(size, True)),
                                    mt.MegaDepthCrop(du.CentralCrop(size, True)),
                                    mt.MegaDepthToTensor()]

            elif mode == l.TEST:
                item_transforms = [mt.MegaDepthToPILImage()]

                if model_config[exp.INPUT_CHANNELS] == 1:
                    item_transforms += [mt.MegaDepthToGrayScale()]

                item_transforms += [mt.MegaDepthResize(du.AspectResize(size, False)),
                                    mt.MegaDepthCrop(du.CentralCrop(size, False)),
                                    mt.MegaDepthToTensor()]

            elif mode == l.ANALYZE:
                item_transforms = [mt.MegaDepthToPILImage()]

                if model_config[exp.INPUT_CHANNELS] == 1:
                    item_transforms += [mt.MegaDepthToGrayScale()]

                item_transforms += [mt.MegaDepthResize(du.AspectResize(size, False)),
                                    mt.MegaDepthCrop(du.CentralCrop(size, False)),
                                    mt.MegaDepthToTensor()]

            else:
                raise NotImplementedError

            dataset = MegaDepthDataset.from_config(d_config, item_transforms)

        elif d_key == du.AACHEN:

            if mode in [l.TEST, l.ANALYZE]:
                item_transforms = [at.AachenToPILImage()]

                if model_config[exp.INPUT_CHANNELS] == 1:
                    item_transforms += [at.AachenToGrayScale()]

                item_transforms += [at.AachenCrop(du.ParityCrop(8)),
                                    at.AachenToTensor()]

                dataset = AachenDataset.from_config(d_config, item_transforms)

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        datasets.append(dataset)

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    else:
        return datasets[0]


def create_loader(dataset, loader_config):
    num_samples = loader_config[du.NUM_SAMPLES]

    batch_size = loader_config[du.BATCH_SIZE]
    shuffle = loader_config[du.SHUFFLE]
    num_workers = loader_config[du.NUM_WORKERS]

    if num_samples == -1:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers)
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=du.DatasetSubsetSampler(dataset, num_samples, shuffle),
                                 num_workers=num_workers)

    return data_loader
