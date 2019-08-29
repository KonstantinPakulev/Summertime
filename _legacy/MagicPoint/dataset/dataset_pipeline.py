import numpy as np
import cv2 as cv
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil
import tempfile

import legacy.MagicPoint.dataset.photometries as photometries
import legacy.MagicPoint.dataset.dataset_generators as dataset_generators

from legacy.MagicPoint.dataset.dataset_generators import set_random_state, generate_background
from legacy.MagicPoint import sample_homography, warp_points, filter_points, compute_valid_mask, flat2mat


def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
        else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def save_primitive_data(primitive, tar_path, config):
    temp_dir = Path(tempfile.gettempdir(), config['name'] + '_{}'.format(config['suffix']), primitive)

    # Clean temp directory before writing into it
    if temp_dir.exists():
        for dir in temp_dir.iterdir():
            shutil.rmtree(dir)

    set_random_state(np.random.RandomState(config['generation']['random_seed']))

    for split, size in config['generation']['split_sizes'].items():
        image_dir, points_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
        image_dir.mkdir(parents=True, exist_ok=True)
        points_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(size), desc=split, leave=False):
            image = generate_background(config['generation']['image_size'],
                                        **config['generation']['params']['generate_background'])
            points = np.array(
                getattr(dataset_generators, primitive)(image, **config['generation']['params'].get(primitive, {})))
            points = np.flip(points, 1)

            b = config['preprocessing']['blur_size']
            image = cv.GaussianBlur(image, (b, b), 0)
            points = (points * np.array(config['preprocessing']['resize'], np.float)
                      / np.array(config['generation']['image_size'], np.float))
            image = cv.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                              interpolation=cv.INTER_LINEAR)

            cv.imwrite(str(Path(image_dir, '{}.png'.format(i))), image)
            np.save(Path(points_dir, '{}.npy'.format(i)), points)

    tar = tarfile.open(tar_path, mode='w:gz')
    tar.add(temp_dir, arcname=primitive)
    tar.close()
    shutil.rmtree(temp_dir)


def photometric_augmentation(image, config):
    primitives = parse_primitives(config['primitives'], photometries.augmentations)
    prim_configs = [config['augmentation']['photometric']['params'].get(p, {}) for p in primitives]

    for p, c in zip(primitives, prim_configs):
        image = getattr(photometries, p)(image, c)

    return image


def homographic_augmentation(image, points, config):

    homography = sample_homography(image.shape, **config['augmentation']['homographic']['params'])

    warped_image = cv.warpPerspective(image, flat2mat(homography)[0], image.shape[::-1], flags=cv.WARP_INVERSE_MAP)

    warped_mask = compute_valid_mask(image.shape, homography, config['augmentation']['homographic']['valid_border_margin'])

    warped_points = warp_points(points, homography)
    warped_points = filter_points(warped_points, warped_image.shape)

    return warped_image, warped_points, warped_mask


def get_keypoint_map(image, points):
    truncated_points = np.minimum(np.round(points), np.array(image.shape) - 1).astype(dtype=np.int32)

    keypoint_map = np.zeros(image.shape)
    keypoint_map[truncated_points[:, 0], truncated_points[:, 1]] = 1

    return keypoint_map