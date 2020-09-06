from abc import ABC, abstractmethod

from torch.optim.lr_scheduler import MultiStepLR

from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.custom_events import Events
from ignite.contrib.handlers.param_scheduler import LRScheduler

from tensorboardX import SummaryWriter

import Net.source.core.loop as l

NET_VGG = "NetVGG"
SUPER_POINT = "SuperPoint"
R2D2 = 'R2D2'

# TODO. Transfer config keys to corresponding files

"""
Model keys
"""
GRID_SIZE = 'grid_size'
DESCRIPTOR_SIZE = 'descriptor_size'
INPUT_CHANNELS = 'input_channels'
BATCH_NORM = 'batch_norm'

NMS_KERNEL_SIZE = 'nms_kernel_size'
SOFT_NMS_KERNEL_SIZE = 'soft_nms_kernel_size'

EXP_NAME = 'exp_name'
CHECKPOINT_NAME = 'checkpoint_name'

"""
Criterion keys
"""
DET = 'det'
DET_CONF = 'det_conf'

DESC = 'desc'
EP = 'ep'
POSE = 'pose'

LOSS_VERSION = 'loss_version'

TOP_K = "top_k"

LAMBDA = 'lambda'
GAUSS_KERNEL_SIZE = "gauss_kernel_size"
GAUSS_SIGMA = "gauss_sigma"

MARGIN = "margin"

"""
Metric keys
"""
LOSS_LOG_ITER = 'loss_log_iter'
METRIC_LOG_ITER = 'metric_log_iter'

LOG_ITER = 'log_iter'
LOG_EPOCH = 'log_epoch'

COLLECT_RATE = 'collect_rate'

PX_THRESH = 'px_thresh'
R_ERR_THRESH = 'r_err_thresh'
T_ERR_THRESH = 't_err_thresh'

DD_MEASURE = 'dd_measure'

METHOD_NAME = 'method_name'


"""
Experiment keys
"""
LR = 'lr'
LR_DECAY = 'lr_decay'
LR_SCHEDULE = 'lr_schedule'
NUM_EPOCHS = 'num_epochs'

RETURN_OUTPUT = 'return_output'

SAVE_INTERVAL = 'save_interval'
N_SAVED = 'n_saved'

"""
Checkpoint and config variables
"""
CHECKPOINT_PREFIX = "my"

MODEL = 'model'
OPTIMIZER = 'optimizer'

CRITERION = 'criterion'
DATASET = 'dataset'
LOADER = 'loader'
METRIC = 'metric'
EXPERIMENT = 'experiment'


class Experiment(ABC):

    def __init__(self, device, mode, model_version, config, log_dir=None, checkpoint_dir=None):
        self.device = device
        self.mode = mode
        self.model_version = model_version

        self.dataset_config = config[DATASET]
        self.loader_config = config[LOADER]

        self.model_config = config[MODEL]
        self.criterion_config = config.get(CRITERION)

        self.metric_config = config.get(METRIC)
        self.experiment_config = config[EXPERIMENT]

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.model = self.get_model()

        if self.mode in [l.TRAIN, l.ANALYZE]:
            self.criterion_chain = self.get_criterion_chain()
        else:
            self.criterion_chain = None

        if self.mode == l.TRAIN:
            self.optimizer_chain = self.get_optimizer_chain()
        else:
            self.optimizer_chain = None

        self.loops = self.get_loops()

        self.writer = None

    @abstractmethod
    def get_model(self):
        ...

    @abstractmethod
    def get_criterion_chain(self):
        ...

    @abstractmethod
    def get_optimizer_chain(self):
        ...

    @abstractmethod
    def get_loops(self):
        ...

    def get_checkpoint_params(self):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoints(self):
        ...

    def bind_checkpoints(self):
        key, score_function = self.get_checkpoint_params()

        checkpoint_saver = ModelCheckpoint(self.checkpoint_dir, CHECKPOINT_PREFIX,
                                           score_function=score_function,
                                           score_name=key,
                                           n_saved=self.experiment_config[N_SAVED],
                                           require_empty=False)

        self.loops[1].engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver,
                                               {MODEL: self.model,
                                                OPTIMIZER: self.optimizer_chain})

    def run(self):
        self.load_checkpoints()

        if self.mode == l.TRAIN:
            if self.checkpoint_dir is not None:
                self.bind_checkpoints()

            if self.log_dir is not None:
                self.writer = SummaryWriter(logdir=self.log_dir)

            lr_schedule = self.experiment_config.get(LR_SCHEDULE)
            if lr_schedule is not None:
                lr_decay = self.experiment_config[LR_DECAY]
                lr_scheduler = MultiStepLR(self.optimizer_chain.optimizer, lr_schedule, lr_decay)

                self.loops[0].engine.add_event_handler(Events.EPOCH_COMPLETED, LRScheduler(lr_scheduler))

        output = self.loops[0].run(self.experiment_config[NUM_EPOCHS],
                                   self.experiment_config[RETURN_OUTPUT])

        if self.writer is not None:
            self.writer.close()

        return output
