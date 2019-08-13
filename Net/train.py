import os
import sys
from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch

from Net.experiments.main_experiment import TrainExperiment
from Net.experiments.other_experiments import DebugTrainExperiment, TrainExperimentAlter, TrainExperimentQHT, TrainExperimentSOSR


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_path", type=str)

    args = parser.parse_args()

    experiment = None

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.exp_id == 'train':
        experiment = TrainExperiment(device, args.log_dir, args.checkpoint_dir)
    elif args.exp_id == 'train_alter':
        experiment = TrainExperimentAlter(device, args.log_dir, args.checkpoint_dir)
    elif args.exp_id == 'train_alter_loss':
        experiment = TrainExperimentSOSR(device, args.log_dir, args.checkpoint_dir)
    else:
        experiment = DebugTrainExperiment(device, args.log_dir, args.checkpoint_dir)

    experiment.run()
