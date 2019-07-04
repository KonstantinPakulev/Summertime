from skimage import io

from MagicPoint.dataset.dataset_pipeline import *
from common.utils import grayscale2rgb, normalize_image

from torch.utils.data import Dataset

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

# Modes of the dataset
TRAINING = 'training'
VALIDATION = 'validation'
TEST = 'test'

# Dataset item dictionary keys
IMAGE = 'image'
POINTS = 'points'
KEYPOINT_MAP = 'keypoint_map'

available_modes = [TRAINING, VALIDATION, TEST]


class ArtificialDataset(Dataset):

    def __init__(self, mode, config):

        assert mode in available_modes

        self.mode = mode
        self.config = config

        primitives = parse_primitives(config['primitives'], primitives_to_draw)

        base_path = Path(config['data_path'], config['name'] + '_{}'.format(config['suffix']))
        base_path.mkdir(parents=True, exist_ok=True)

        self.images = []
        self.points = []

        for primitive in primitives:
            tar_path = Path(base_path, '{}.tag.gz'.format(primitive))
            if not tar_path.exists():
                save_primitive_data(primitive, tar_path, config)

            tar = tarfile.open(tar_path)
            temp_dir = Path(tempfile.gettempdir())
            tar.extractall(path=temp_dir)
            tar.close()

            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)

            e = [str(p) for p in Path(path, 'images', self.mode).iterdir()]
            f = [p.replace('images', 'points') for p in e]
            f = [p.replace('.png', '.npy') for p in f]

            self.images.extend(e[:int(truncate * len(e))])
            self.points.extend(f[:int(truncate * len(f))])

    def __len__(self):
       return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        points_path = self.points[index]

        image = np.asarray(io.imread(image_path)).astype(np.float32)
        points = np.load(points_path).astype(np.float32)

        # Apply data augmentation
        if self.mode == 'training':
            if self.config['augmentation']['photometric']['enable']:
                image = photometric_augmentation(image, self.config)
            if self.config['augmentation']['homographic']['enable']:
                image, points = homographic_augmentation(image, points, self.config)

        # Convert points to keypoint map
        keypoint_map = get_keypoint_map(image, points)
        image = normalize_image(grayscale2rgb(image))

        item = {}
        item[IMAGE] = image
        item[POINTS] = points
        item[KEYPOINT_MAP] = keypoint_map

        return item
