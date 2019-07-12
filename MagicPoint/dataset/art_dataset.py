from skimage import io

from MagicPoint.dataset.dataset_pipeline import *

from common.utils import *

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

primitives_to_draw = [
    'draw_lines',
    'draw_polygon',
    'draw_multiple_polygons',
    'draw_ellipses',
    'draw_star',
    'draw_checkerboard',
    'draw_stripes',
    'draw_cube',
    'gaussian_noise'
]

available_modes = [TRAINING, VALIDATION, TEST]


def collate(batch):
    return {IMAGE: default_collate([d[IMAGE] for d in batch]).float(),
            KEYPOINT_MAP: default_collate([d[KEYPOINT_MAP] for d in batch]).float(),
            MASK: default_collate([d[MASK] for d in batch]).float().unsqueeze(0)}


class ArtificialDataset(Dataset):

    def __init__(self, mode, config):

        assert mode in available_modes

        self.mode = mode
        self.config = config

        primitives = parse_primitives(config['primitives'], primitives_to_draw)

        base_path = Path(config['data_path'], config['name'] + '_{}'.format(config['suffix']))
        base_path.mkdir(parents=True, exist_ok=True)

        # print("Creating base path:", base_path)

        self.images = []
        self.points = []

        for primitive in primitives:
            tar_path = Path(base_path, '{}.tag.gz'.format(primitive))
            # print("Tar file location:", tar_path)

            if not tar_path.exists():
                save_primitive_data(primitive, tar_path, config)

            temp_dir = Path(tempfile.gettempdir(), config['name'] + '_{}'.format(config['suffix']))
            temp_dir.mkdir(parents=True, exist_ok=True)

            # print("Reserving temp dir:", temp_dir)

            tar = tarfile.open(tar_path)
            tar.extractall(path=temp_dir)
            tar.close()

            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)

            e = [str(p) for p in Path(path, 'images', self.mode).iterdir()]

            # print(len(e))

            f = [p.replace('images', 'points') for p in e]
            f = [p.replace('.png', '.npy') for p in f]

            self.images.extend(e[:int(truncate * len(e))])
            self.points.extend(f[:int(truncate * len(f))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        points_path = self.points[index]

        # image in form (h, w)
        image = np.asarray(io.imread(image_path))
        # array in form (n, 2)
        points = np.load(points_path)
        # image in form (h, w)
        mask = np.ones(image.shape)

        # Apply data augmentation
        if self.mode == 'training':
            if self.config['augmentation']['photometric']['enable']:
                image = photometric_augmentation(image, self.config)
            if self.config['augmentation']['homographic']['enable']:
                image, points, mask = homographic_augmentation(image, points, self.config)

        # Convert points to keypoint map
        keypoint_map = get_keypoint_map(image, points)
        image = normalize_image(grayscale2rgb(image))

        item = {IMAGE: image, POINTS: points, KEYPOINT_MAP: keypoint_map, MASK: mask}

        return item
