import os
import cv2

import numpy as np
import pandas as pd

import Net.source.datasets.dataset_utils as du


"""
MegaDepth dataset annotation and creation utils
"""


def create_dataset_annotations(dataset_root, scene_info_root):
    annotations_dict = {du.SCENE_NAME: [],

                        du.IMAGE1: [],
                        du.IMAGE2: [],

                        du.DEPTH1: [],
                        du.DEPTH2: [],

                        du.ID1: [],
                        du.ID2: []}

    min_overlap_ratio = 0.5
    max_overlap_ratio = 1.0
    max_scale_ratio = np.inf

    scene_files = os.listdir(scene_info_root)
    scene_files = [name for name in scene_files if name.endswith(".npz")]

    for scene_file in scene_files:
        scene_path = os.path.join(scene_info_root, scene_file)
        scene_info = np.load(scene_path, allow_pickle=True)

        # Square matrix of overlap between two images
        overlap_matrix = scene_info['overlap_matrix']
        # Square matrix of depth ratio between two estimated depths
        scale_ratio_matrix = scene_info['scale_ratio_matrix']

        valid = np.logical_and(np.logical_and(overlap_matrix >= min_overlap_ratio, overlap_matrix <= max_overlap_ratio),
                               scale_ratio_matrix <= max_scale_ratio)
        # Pairs of overlapping images
        pairs = np.vstack(np.where(valid)).transpose(1, 0)

        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']

        for pair in pairs:
            # Pair of matching images
            idx1, idx2 = pair

            scene_name = scene_file.split('.')[0]

            image_path1, image_path2 = os.path.join(dataset_root, image_paths[idx1]), \
                                       os.path.join(dataset_root, image_paths[idx2])

            depth_path1, depth_path2 = os.path.join(dataset_root, depth_paths[idx1]), \
                                       os.path.join(dataset_root, depth_paths[idx2])

            annotations_dict[du.SCENE_NAME].append(scene_name)

            annotations_dict[du.IMAGE1].append(image_path1)
            annotations_dict[du.IMAGE2].append(image_path2)

            annotations_dict[du.DEPTH1].append(depth_path1)
            annotations_dict[du.DEPTH2].append(depth_path2)

            annotations_dict[du.ID1].append(idx1)
            annotations_dict[du.ID2].append(idx2)

    annotations = pd.DataFrame(data=annotations_dict)
    annotations.to_csv(os.path.join(scene_info_root, 'annotations.csv'))


def create_annotations_by_scene(scene_info_root, mode, dataset_name, scenes, is_in=True):
    annotations = pd.read_csv(os.path.join(scene_info_root, 'annotations.csv'), index_col=[0])

    if is_in:
        mode_annotations = annotations.loc[annotations[du.SCENE_NAME].isin(scenes)]
    else:
        mode_annotations = annotations.loc[~annotations[du.SCENE_NAME].isin(scenes)]

    mode_annotations.to_csv(os.path.join(scene_info_root, f'{mode}_{dataset_name}.csv'))


def create_annotations_for_warp(csv_path):
    warp_annotations = pd.read_csv(csv_path, index_col=[0]).drop([du.ID1, du.ID2], axis=1)

    warp_annotations1 = warp_annotations.drop_duplicates(du.IMAGE1).drop(du.IMAGE2, axis=1)
    warp_annotations2 = warp_annotations.drop_duplicates(du.IMAGE2).drop(du.IMAGE1, axis=1).rename(columns={du.IMAGE2: du.IMAGE1})
    warp_annotations = pd.concat([warp_annotations1, warp_annotations2], sort=False).drop_duplicates(du.IMAGE1)

    split = csv_path.split('/')
    csv_name = split[-1].split('.')[-2]
    scene_info_root = '/'.join(split[:-1])

    warp_annotations.to_csv(os.path.join(scene_info_root, f'{csv_name}_warp.csv'))


def create_sampled_annotations_by_scene(scene_info_root, mode, dataset_name, scenes, num_samples):
    annotations = pd.read_csv(os.path.join(scene_info_root, 'annotations.csv'), index_col=[0])

    mode_annotations = annotations.loc[annotations[du.SCENE_NAME].isin(scenes)]
    mode_annotations = mode_annotations.groupby(du.SCENE_NAME, group_keys=False).apply(pd.DataFrame.sample, n=num_samples)

    mode_annotations.to_csv(os.path.join(scene_info_root, f'{mode}_{dataset_name}.csv'))


def create_annotations_by_log(log, annotations_path, log_path, file_name):
    annotations = pd.read_csv(annotations_path, index_col=[0])

    scene = prepare_scene_name(log[du.SCENE_NAME]).unique().tolist()
    annotations = annotations[annotations[du.SCENE_NAME].isin(scene)]

    feature = annotations[du.ID1].astype(str) + annotations[du.ID2].astype(str)
    selection = (log[du.ID1].astype(str) + log[du.ID2].astype(str)).tolist()

    file_path = "/".join(log_path.split("/")[:-1])
    file_path = os.path.join(file_path, f"{file_name}.csv")

    annotations = annotations[feature.isin(selection)]
    annotations.to_csv(file_path)


"""
Dataset visualization utils
"""


def visualize_image_pair(pair_info, annotations_path):
    annotations = pd.read_csv(annotations_path, index_col=[0])

    # TODO.


"""
Support utils
"""


def prepare_scene_name(scene_name):
    return scene_name.astype(str).apply(lambda x: x.zfill(4))
