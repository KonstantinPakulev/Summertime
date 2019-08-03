from easydict import EasyDict

cfg = EasyDict()

"""
Loss settings
"""
cfg.LOSS = EasyDict()

cfg.LOSS.NMS_THRESH = 0.0
cfg.LOSS.NMS_K_SIZE = 5

cfg.LOSS.TOP_K = 512

"""
Dataset settings
"""
cfg.DATASET = EasyDict()

cfg.DATASET.view = EasyDict()
cfg.DATASET.view.root = "../../data/hpatch_v_sequence"
cfg.DATASET.view.csv = "hpatch_view.csv"
cfg.DATASET.view.MEAN = 0.4230204841414801
cfg.DATASET.view.STD = 0.25000138349993173

cfg.DATASET.view.analyze_csv = "analyze.csv"
