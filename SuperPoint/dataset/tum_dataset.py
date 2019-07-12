from skimage import io

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from MagicPoint.dataset.homographies import *

from common.utils import *


def collate(batch):
    return {IMAGE: default_collate([d[IMAGE] for d in batch]).float(), KEYPOINT_MAP: default_collate([d[KEYPOINT_MAP] for d in batch]).float(),
            WARPED_IMAGE: default_collate([d[WARPED_IMAGE] for d in batch]).float(), WARPED_KEYPOINT_MAP: default_collate([d[WARPED_KEYPOINT_MAP] for d in batch]).float(),
            MASK: default_collate([d[MASK] for d in batch]).unsqueeze(0).transpose(0, 1).float(),
            WARPED_MASK: default_collate([d[WARPED_MASK] for d in batch]).unsqueeze(0).transpose(0, 1).float(),
            HOMOGRAPHY: default_collate([d[HOMOGRAPHY] for d in batch]).float(),
            DEPTH: default_collate([d[DEPTH] for d in batch]).float()}


class TUMDataset(Dataset):

    def __init__(self, mode, config, train_ratio=None):
        self.config = config
        self.base_path = os.path.join(config['data_path'], config['name'])

        self.images = read_tum_list(self.base_path, 'rgb.txt')
        self.keypoint_maps = read_tum_list(self.base_path, 'keypoint_map.txt')
        self.depths = read_tum_list(self.base_path, 'depth.txt')

        if mode == VALIDATION:
            l = int(len(self.images) * train_ratio)
            self.images = self.images[:l]
            self.keypoint_maps = self.keypoint_maps[:l]
            self.depths = self.depths[:l]

    def __getitem__(self, item):
        image_path = os.path.join(self.base_path, self.images[item])
        keypoint_map_path = os.path.join(self.base_path, self.keypoint_maps[item])
        depth_path = os.path.join(self.base_path, self.depths[item])

        image = normalize_image(np.asarray(io.imread(image_path))[..., [2, 1, 0]]).transpose((2, 0, 1))
        keypoint_map = normalize_image(np.asarray(io.imread(keypoint_map_path)))

        homography = sample_homography(image.shape[1:], self.config['warped_pair']['params'])
        mat_homography = flat2mat(homography)[0]

        warped_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            warped_image[i] = cv.warpPerspective(image[i], mat_homography, image.shape[::-1][:2], flags=cv.WARP_INVERSE_MAP)

        warped_keypoint_map = cv.warpPerspective(keypoint_map, mat_homography, keypoint_map.shape[::-1], flags=cv.WARP_INVERSE_MAP)

        mask = np.ones(image.shape[1:], dtype=np.float)
        warped_mask = compute_valid_mask(image.shape[1:], homography, self.config['warped_pair']['valid_border_margin'])

        # TODO. Options: normalize by max value, normalize by collected statistics
        depth = np.asarray(io.imread(depth_path)).astype(np.float)
        depth /= depth.ravel().max()

        return {IMAGE: image, KEYPOINT_MAP: keypoint_map,
                WARPED_IMAGE: warped_image, WARPED_KEYPOINT_MAP: warped_keypoint_map,
                MASK: mask, WARPED_MASK: warped_mask,
                HOMOGRAPHY: homography,
                DEPTH: depth}

    def __len__(self):
        return len(self.images)
