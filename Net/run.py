import os
import sys
from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch

from Net.experiments.experiment import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_iter", type=str)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    experiment = getattr(sys.modules[__name__], args.exp_id)(device, args.log_dir, args.checkpoint_dir, args.checkpoint_iter)
    experiment.run()
