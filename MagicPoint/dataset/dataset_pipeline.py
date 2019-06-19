import math
import numpy as np
import cv2 as cv
import sys
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil
import tempfile

import MagicPoint.dataset.dataset_generators as dataset_generators
from MagicPoint.dataset.dataset_generators import set_random_state, generate_background


def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
        else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def save_primitive_data(primitive, tar_path, config):
    temp_dir = Path(tempfile.gettempdir(), primitive)

    set_random_state(np.random.RandomState(config['generation']['random_seed']))

    for split, size in config['generation']['split_sizes'].items():
        image_dir, points_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
        image_dir.mkdir(parents=True, exist_ok=True)
        points_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(size), desc=split, leave=False):
            image = generate_background(config['generation']['image_size'],
                                        **config['generation']['params']['generate_background'])
            points = np.array(getattr(dataset_generators, primitive)(image, **config['generation']['params'].get(primitive, {})))
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


def photometric_augmentation(image, points, config):
    # with tf.name_scope('photometric_augmentation'):
    #     primitives = parse_primitives(config['primitives'], photaug.augmentations)
    #     prim_configs = [config['params'].get(
    #         p, {}) for p in primitives]
    #
    #     indices = tf.range(len(primitives))
    #     if config['random_order']:
    #         indices = tf.random_shuffle(indices)
    #
    #     def step(i, image):
    #         fn_pairs = [(tf.equal(indices[i], j),
    #                      lambda p=p, c=c: getattr(photaug, p)(image, **c))
    #                     for j, (p, c) in enumerate(zip(primitives, prim_configs))]
    #         image = tf.case(fn_pairs)
    #         return i + 1, image
    #
    #     _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
    #                              step, [0, data['image']], parallel_iterations=1)
    #
    # return {**data, 'image': image}
    pass


def homographic_augmentation(image, points, config):
    # with tf.name_scope('homographic_augmentation'):
    #     image_shape = tf.shape(data['image'])[:2]
    #     homography = sample_homography(image_shape, **config['params'])[0]
    #     warped_image = tf.contrib.image.transform(
    #             data['image'], homography, interpolation='BILINEAR')
    #     valid_mask = compute_valid_mask(image_shape, homography,
    #                                     config['valid_border_margin'])
    #
    #     warped_points = warp_points(data['keypoints'], homography)
    #     warped_points = filter_points(warped_points, image_shape)
    #
    # ret = {**data, 'image': warped_image, 'keypoints': warped_points,
    #        'valid_mask': valid_mask}
    # if add_homography:
    #     ret['homography'] = homography
    # return ret
    pass



def get_keypoint_map(image, points):
    truncated_points = np.minimum(np.round(points), np.array(image.shape) - 1).astype(dtype=np.int32)

    keypoint_map = np.zeros(image.shape)
    keypoint_map[truncated_points[:,0], truncated_points[:, 1]] = 1

    return keypoint_map

