import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import skimage.io

from torch.utils.data import DataLoader

from legacy.MagicPoint.dataset.tum_dataset import *
from legacy.MagicPoint.model.magic_point import MagicPoint

from legacy.common.utils import *
from legacy.common.model_utils import homography_adaptation


def label():
    config = load_config('configs/tum_config.yaml')

    data_config = config['data']
    model_config = config['model']
    experiment_config = config['experiment']

    set_seed(experiment_config['seed'])

    dataset = TUMDataset(data_config)
    dataset_loader = DataLoader(dataset, model_config['batch_size'], num_workers=8, collate_fn=collate)

    # device = torch.device('cpu')
    device = torch.device(type='cuda', index=int(os.environ["CUDA_VISIBLE_DEVICES"]))

    model = MagicPoint(model_config).to(device)

    if experiment_config['load_checkpoints']:
        checkpoint_path = get_checkpoint_path(experiment_config, model_config,
                                              experiment_config['load_checkpoint_iter'])
        if checkpoint_path.exists():
            epoch, model_sd, optimizer_sd = load_checkpoint(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(model_sd)

    keypoint_maps_dir = os.path.join(dataset.base_path, 'keypoint_maps')
    if not os.path.exists(keypoint_maps_dir):
        os.mkdir(keypoint_maps_dir)

    keypoint_map = []

    with torch.no_grad():
        for item in dataset_loader:
            path = os.path.join(keypoint_maps_dir, item[NAME][0] + ".png")
            keypoint_map.append(item[NAME][0] + " " + os.path.join('keypoint_maps', item[NAME][0] + ".png"))

            if not os.path.exists(path) or experiment_config['rewrite']:
                y_pred = model(item[IMAGE].to(device))
                probs = homography_adaptation(item[IMAGE].to(device), y_pred['probs'], model, device, model_config)

                probs = probs.detach().cpu().numpy()
                probs[np.where(probs)] = 1

                skimage.io.imsave(path, to255scale(probs))

    with open(os.path.join(dataset.base_path, 'keypoint_map.txt'), 'w') as file:
        file.write('# keypoint maps\n')
        file.write('# file: none\n')
        file.write('# timestamp filename\n')

        for item in keypoint_map:
            file.write("{}\n".format(item))

    os.rename(dataset.base_path, os.path.join(data_config['dest_path'], data_config['name']))

if __name__ == "__main__":
    label()
