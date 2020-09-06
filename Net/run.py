import os
import sys

from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from Net.source.utils.run_utils import create_experiment

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--mode_config_path", type=str)

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_version", type=str)

    parser.add_argument("--mode", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--gpu", type=str)

    args = parser.parse_args()

    experiment = create_experiment(args.model_config_path, args.mode_config_path, args.model_name, args.model_version,
                                   args.mode, args.dataset_name, args.gpu)

    experiment.run()

