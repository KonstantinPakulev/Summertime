import os
from torch.utils.data import DataLoader

from MagicPoint.model.magic_point import MagicPoint
from MagicPoint.dataset.artificial_dataset import ArtificialDataset
from common.utils import load_config, set_seed


def main():
    config =  load_config('main_config.yaml')
    data_config = config['data']
    model_config = config['model']

    set_seed(config['seed'])
    train_iter = config['train_iter']

    dataset = ArtificialDataset('training', data_config)
    data_loader = DataLoader(dataset, model_config['batch_size'], shuffle=True, num_workers=4)

    # for i in range(train_iter):
    #     for batch in data_loader:
    #         print(batch.shape)
    #         break


if __name__ == "__main__":
    main()
