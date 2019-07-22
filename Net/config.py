from easydict import EasyDict

cfg = EasyDict()

"""
Model settings
"""
cfg.MODEL = EasyDict()
cfg.MODEL.GRID_SIZE = 8
cfg.MODEL.DESCRIPTOR_SIZE = 16

"""
Loss settings
"""
cfg.LOSS = EasyDict()
cfg.LOSS.POS_LAMBDA = 800
cfg.LOSS.POS_MARGIN = 1
cfg.LOSS.NEG_MARGIN = 0.2

"""
Train settings
"""
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_EPOCHS = 300
cfg.TRAIN.LOG_INTERVAL = 2
cfg.TRAIN.WEIGHT_DECAY = 0

"""
Val settings
"""
cfg.VAL = EasyDict()
cfg.VAL.BATCH_SIZE = 1

"""
Test settings
"""
cfg.TEST = EasyDict()
cfg.TEST.BATCH_SIZE = 1

"""
Dataset settings
"""
cfg.DATASET = EasyDict()
cfg.DATASET.SPLIT = [0.8, 0.1, 0.1]

cfg.DATASET.view = EasyDict()
cfg.DATASET.view.root = "../data/hpatch_v_sequence"
cfg.DATASET.view.csv = "hpatch_view.csv"
cfg.DATASET.view.MEAN = 0.4230204841414801
cfg.DATASET.view.STD = 0.25000138349993173

cfg.DATASET.TRAIN = cfg.DATASET.view
cfg.DATASET.VAL = cfg.DATASET.view


