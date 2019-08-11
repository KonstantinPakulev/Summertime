from easydict import EasyDict

cfg = EasyDict()

"""
Model settings
"""
cfg.MODEL = EasyDict()
cfg.MODEL.GRID_SIZE = 8
cfg.MODEL.DESCRIPTOR_SIZE = 32

"""
Loss settings
"""
cfg.LOSS = EasyDict()

"""
Triplet loss
"""
cfg.LOSS.DES_LAMBDA = 1
cfg.LOSS.MARGIN = 1

"""
MSE loss
"""
cfg.LOSS.DET_LAMBDA = 100000

cfg.LOSS.NMS_THRESH = 0.0
cfg.LOSS.NMS_K_SIZE = 5

cfg.LOSS.TOP_K = 512

cfg.LOSS.GAUSS_K_SIZE = 15
cfg.LOSS.GAUSS_SIGMA = 0.5

"""
Metric settings
"""
cfg.METRIC = EasyDict()
cfg.METRIC.DET_THRESH = 5.0
cfg.METRIC.DES_THRESH = 1.0
cfg.METRIC.DES_RATIO = 0.7

"""
Train settings
"""
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_EPOCHS = 3000
cfg.TRAIN.LOG_INTERVAL = 2
cfg.TRAIN.LR = 0.001

"""
Val settings
"""
cfg.VAL = EasyDict()
cfg.VAL.BATCH_SIZE = 32
cfg.VAL.LOG_INTERVAL = 8

"""
Show settings
"""
cfg.VAL_SHOW = EasyDict()
cfg.VAL_SHOW.BATCH_SIZE = 1
cfg.VAL_SHOW.LOG_INTERVAL = 5

"""
Dataset settings
"""
cfg.DATASET = EasyDict()

cfg.DATASET.view = EasyDict()
cfg.DATASET.view.root = "../data/hpatch_v_sequence"
cfg.DATASET.view.csv = "hpatch_view.csv"
cfg.DATASET.view.MEAN = 0.4230204841414801
cfg.DATASET.view.STD = 0.25000138349993173

cfg.DATASET.view.train_csv = "train.csv"
cfg.DATASET.view.val_csv = "val.csv"
cfg.DATASET.view.show_csv = "show.csv"
