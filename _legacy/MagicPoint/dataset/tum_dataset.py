from skimage import io

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from legacy.common.utils import *


def collate(batch):
    return {NAME: default_collate([d[NAME] for d in batch]),
            IMAGE: default_collate([d[IMAGE] for d in batch]).float(),
            DEPTH: default_collate([d[DEPTH] for d in batch]).float()}


class TUMDataset(Dataset):

    def __init__(self, config):
        self.base_path = os.path.join(config['data_path'], config['name'])

        self.images = read_tum_list(self.base_path, 'rgb.txt')
        self.depths = read_tum_list(self.base_path, 'depth.txt')

    def __getitem__(self, item):
        image_path = os.path.join(self.base_path, self.images[item])
        depth_path = os.path.join(self.base_path, self.depths[item])

        name = os.path.splitext(self.images[item].split('/')[-1])[0]

        # Correct order of channels and make shape in form of (c, h, w)
        image = normalize_image(np.asarray(io.imread(image_path))[..., [2, 1, 0]]).transpose((2, 0, 1))
        depth = np.asarray(io.imread(depth_path)).astype(np.float)
        depth /= depth.ravel().max()

        return {NAME: name, IMAGE: image, DEPTH: depth}

    def __len__(self):
        return len(self.images)