import os
import re
import yaml
import shutil

import torch

import Net.source.core.loop as l

from Net.source.core import experiment as exp

from Net.source.experiments.net import NetVGGExperiment


def create_experiment(model_config_path, mode_config_path, model_name, model_version,
                      mode, dataset_name, gpu, relative_path=''):
    model_config = load_config(model_config_path)
    mode_config = load_config(mode_config_path)

    model_config[exp.MODEL] = resolve_version_content(model_config[exp.MODEL], model_name, model_version)

    if mode != l.TEST:
        model_config[exp.CRITERION] = resolve_version_content(model_config[exp.CRITERION], model_name, model_version)

    else:
        del model_config[exp.CRITERION]

    if exp.MODEL in mode_config:
        mode_config[exp.MODEL] = resolve_version_content(mode_config[exp.MODEL], model_name, model_version)

    if mode in [l.TEST, l.ANALYZE]:
        mode_config[exp.DATASET] = prepare_dataset_config(mode_config[exp.DATASET], mode, dataset_name)

    log_dir, checkpoint_dir = prepare_directories(mode_config, mode, model_name, model_version, relative_path)

    device = prepare_device(gpu)

    merge_dict(model_config, mode_config)

    print_dict(model_config)

    if mode != l.ANALYZE:
        with open(f'{log_dir}/{mode}_config.yaml', 'w') as file:
            yaml.dump(model_config, file)

    return NetVGGExperiment(device, mode, model_version, model_config, log_dir, checkpoint_dir)


"""
Support utils
"""


def prepare_device(gpu):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_config(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
        return config


def resolve_version_content(config, model_name, model_version):
    sub_config = config[model_name]

    to_merge = []
    to_remove = []

    for key in sub_config.keys():
        if re.search(r"v\d*", key):
            if model_version != '' and re.search(rf"{model_version}($|_)", key):
                to_merge.append(key)
            else:
                to_remove.append(key)

    for key in to_merge:
        merge_dict(sub_config, sub_config[key])

    to_remove.extend(to_merge)

    for key in to_remove:
        del sub_config[key]

    return sub_config


def merge_dict(dict_a, dict_b):
    for key, value in dict_b.items():
        if isinstance(value, dict):
            if key in dict_a:
                merge_dict(dict_a[key], value)
            else:
                dict_a[key] = value
        else:
            dict_a[key] = value


def prepare_dataset_config(dataset_config, mode, dataset_name):
    sub_config = dataset_config[mode]

    to_remove = []

    datasets = dataset_name.split(',')

    for key in sub_config.keys():
        if key not in datasets:
            to_remove.append(key)

    for key in to_remove:
        del sub_config[key]

    return dataset_config


def prepare_directories(config, mode, model_name, model_version, relative_path):
    if model_version != '':
        model_name = f"{model_name}_{model_version}"

    log_dir = os.path.join(relative_path, 'runs', model_name)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')

    if exp.MODEL not in config or not isinstance(config[exp.MODEL], dict) or exp.CHECKPOINT_NAME not in config[exp.MODEL]:
        if mode == l.TRAIN:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)

            os.mkdir(log_dir)
            os.mkdir(checkpoint_dir)

    return log_dir, checkpoint_dir


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f"{key}")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            if isinstance(value[0], dict):
                print("\t" * indent + f"{key}")
                for elem in value:
                    print_dict(elem, indent + 1)
            else:
                print("\t" * indent + f"{key:>18} : {value}")
        else:
            print("\t" * indent + f"{key:>18} : {value}")
