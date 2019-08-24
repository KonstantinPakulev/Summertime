import os
import sys
from argparse import ArgumentParser

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch

from Net.experiments.base.config import DebugConfig

from Net.experiments.main_experiment import TE, TED
from Net.experiments.rf_experiments import TEDRF, TEDDiffRF
from Net.experiments.main_alter_experiment import TERSOSR, TENoSOSR, TERFOSNoSOSR, TECRFOSNoSOSR

class DTE(TECRFOSNoSOSR):

    def init_config(self):
        self.config = DebugConfig()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_path", type=str)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    experiment = getattr(sys.modules[__name__], args.exp_id)(device, args.log_dir, args.checkpoint_dir)
    experiment.run()
