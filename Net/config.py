from easydict import EasyDict

cfg = EasyDict()

"""
Model settings
"""
cfg.MODEL = EasyDict()
cfg.MODEL.GRID_SIZE = 8
cfg.MODEL.DESCRIPTOR_SIZE = 128

"""
Loss settings
"""
cfg.LOSS = EasyDict()
# cfg.LOSS.POS_LAMBDA = 770.4600830078125
cfg.LOSS.POS_LAMBDA = 1577.845062256
cfg.LOSS.POS_MARGIN = 1
cfg.LOSS.NEG_MARGIN = 0.2

"""
Train settings
"""
cfg.TRAIN = EasyDict()
# cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.BATCH_SIZE = 2
# cfg.TRAIN.NUM_EPOCHS = 300
cfg.TRAIN.NUM_EPOCHS = 5000000
cfg.TRAIN.LOG_INTERVAL = 2
cfg.TRAIN.LR = 0.001
cfg.TRAIN.WEIGHT_DECAY = 1e-6
cfg.TRAIN.SCH_STEP = [4000, 10000]
cfg.TRAIN.SCH_GAMMA = 0.2

"""
Val settings
"""
cfg.VAL = EasyDict()
cfg.VAL.BATCH_SIZE = 8

"""
Test settings
"""
cfg.TEST = EasyDict()
cfg.TEST.BATCH_SIZE = 1

"""
Dataset settings
"""
cfg.DATASET = EasyDict()
# cfg.DATASET.SPLIT = [0.8, 0.1, 0.1]
cfg.DATASET.SPLIT = [0.01, 0.1, 0.1]

cfg.DATASET.view = EasyDict()
cfg.DATASET.view.root = "../data/hpatch_v_sequence"
cfg.DATASET.view.csv = "hpatch_view.csv"
cfg.DATASET.view.MEAN = 0.4230204841414801
cfg.DATASET.view.STD = 0.25000138349993173

cfg.DATASET.TRAIN = cfg.DATASET.view
cfg.DATASET.VAL = cfg.DATASET.view
