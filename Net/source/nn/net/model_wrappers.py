import Net.source.core.experiment as exp
import Net.source.core.loop as l
import Net.source.datasets.dataset_utils as du
import Net.source.nn.net.utils.endpoint_utils as eu
import Net.source.utils.metric_utils as meu

from Net.source.core.wrapper import ModuleWrapper, ModelContainer
from Net.source.core.ignite_metrics import AveragePeriodicMetric, DetailedMetric

from Net.source.utils.ignite_utils import RepTransformer, MSTransformer, MMATransformer, EMSTransformer, \
    PoseTransformer, ParamPoseTransformer

from Net.source.utils.matching_utils import DescriptorDistance

from Net.source.nn.net.models.det_models import DetJointBranch
from Net.source.nn.net.models.desc_models import DescBranch
from Net.source.nn.net.models.loc_models import LocBranch

from Net.source.nn.net.utils.endpoint_utils import select_kp, sample_descriptors, sample_loc, warp_points

X1 = 'x1'
X2 = 'x2'


class NetContainer(ModelContainer):

    def attach(self, engine, bundle):
        super().attach(engine, bundle)
        if bundle[l.MODE] in [l.TRAIN, l.VAL]:
            px_thresh = bundle[exp.PX_THRESH]
            metric_log_iter = bundle.get(exp.METRIC_LOG_ITER)

            AveragePeriodicMetric(RepTransformer(px_thresh), metric_log_iter).attach(engine, meu.REP)
            AveragePeriodicMetric(MSTransformer(px_thresh, DescriptorDistance.INV_COS_SIM), metric_log_iter).attach(
                engine, meu.MS)
            AveragePeriodicMetric(MMATransformer(px_thresh, DescriptorDistance.INV_COS_SIM), metric_log_iter).attach(
                engine, meu.MMA)

            if bundle[l.MODE] == l.VAL:

                AveragePeriodicMetric(EMSTransformer(px_thresh, self.device)).attach(engine, meu.EMS)

        elif bundle[l.MODE] == l.TEST:
            if du.MEGADEPTH in bundle[du.DATASET_NAME]:
                px_thresh = bundle[exp.PX_THRESH]
                num_cat = len(px_thresh)

                DetailedMetric(RepTransformer(px_thresh, True), len(px_thresh)).attach(engine, meu.REP)
                DetailedMetric(MSTransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), num_cat).attach(
                    engine, meu.MS)
                DetailedMetric(MMATransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), num_cat).attach(
                    engine, meu.MMA)
                DetailedMetric(EMSTransformer(px_thresh, self.device, True), num_cat).attach(engine, meu.EMS)

                DetailedMetric(PoseTransformer(px_thresh, self.device), num_cat).attach(engine, meu.REL_POSE)
                DetailedMetric(ParamPoseTransformer(px_thresh, self.device), num_cat).attach(engine, meu.PARAM_REL_POSE)

            elif du.HPATCHES_VIEW in bundle[du.DATASET_NAME] or du.HPATCHES_ILLUM in bundle[du.DATASET_NAME]:
                px_thresh = bundle[exp.PX_THRESH]
                num_cat = len(px_thresh)

                DetailedMetric(RepTransformer(px_thresh, True), len(px_thresh)).attach(engine, meu.REP)
                DetailedMetric(MSTransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), num_cat).attach(
                    engine, meu.MS)
                DetailedMetric(MMATransformer(px_thresh, DescriptorDistance.INV_COS_SIM, True), num_cat).attach(
                    engine, meu.MMA)


class BackboneWrapper(ModuleWrapper):

    def __init__(self, device, model_config, backbone):
        super().__init__(device)
        self.backbone = backbone

        self.deploy_mode = model_config.get(exp.DEPLOY_MODE)

    def forward(self, engine, batch, endpoint, bundle):
        if self.deploy_mode is not None and self.deploy_mode:
            x1 = self.backbone(batch[du.IMAGE1].to(self.device))

            bundle[X1] = x1

        else:
            x1 = self.backbone(batch[du.IMAGE1].to(self.device))
            x2 = self.backbone(batch[du.IMAGE2].to(self.device))

            bundle[X1] = x1
            bundle[X2] = x2

        return endpoint, bundle


class DetJointBranchWrapper(ModuleWrapper):

    def __init__(self, device, model_config):
        super().__init__(device)
        self.det = DetJointBranch.from_config(model_config)

        self.deploy_mode = model_config.get(exp.DEPLOY_MODE)
        self.nms_kernel_size = model_config[exp.NMS_KERNEL_SIZE]
        self.top_k = model_config.get(exp.TOP_K)

    def forward(self, engine, batch, endpoint, bundle):
        if self.deploy_mode is not None and self.deploy_mode:
            x1 = bundle[X1]

            score1, _, _, _ = self.det(x1[0], x1[3])

            kp1 = select_kp(score1, self.nms_kernel_size, self.top_k, scale_factor=2.0)

            endpoint[eu.KP1] = kp1

        else:
            x1, x2 = bundle[X1], bundle[X2]
            image1, image2 = batch[du.IMAGE1], batch[du.IMAGE2]

            score1, conf_score1, log_conf_score1, sal_score1 = self.det(x1[0], x1[3])
            score2, conf_score2, log_conf_score2, sal_score2 = self.det(x2[0], x2[3])

            kp1 = select_kp(score1, self.nms_kernel_size, self.top_k, scale_factor=2.0)
            kp2 = select_kp(score2, self.nms_kernel_size, self.top_k, scale_factor=2.0)

            w_kp1, w_vis_kp1_mask, w_kp2, w_vis_kp2_mask = warp_points(kp1, kp2, image1.shape, image2.shape, batch)

            endpoint[eu.SCORE1] = score1
            endpoint[eu.SCORE2] = score2

            endpoint[eu.SAL_SCORE1] = sal_score1
            endpoint[eu.SAL_SCORE2] = sal_score2

            endpoint[eu.CONF_SCORE1] = conf_score1
            endpoint[eu.CONF_SCORE2] = conf_score2

            endpoint[eu.LOG_CONF_SCORE1] = log_conf_score1
            endpoint[eu.LOG_CONF_SCORE2] = log_conf_score2

            endpoint[eu.KP1] = kp1
            endpoint[eu.KP2] = kp2

            endpoint[eu.W_KP1] = w_kp1
            endpoint[eu.W_KP2] = w_kp2

            endpoint[eu.W_VIS_KP1_MASK] = w_vis_kp1_mask
            endpoint[eu.W_VIS_KP2_MASK] = w_vis_kp2_mask

        return endpoint, bundle


class LocJointBranchWrapper(ModuleWrapper):

    def __init__(self, device, model_config):
        super().__init__(device)
        self.loc = LocBranch.from_config(model_config)

        self.deploy_mode = model_config.get(exp.DEPLOY_MODE)

    def forward(self, engine, batch, endpoint, bundle):
        if self.deploy_mode is not None and self.deploy_mode:
            x1 = bundle[X1]
            kp1 = endpoint[eu.KP1]

            loc1 = self.loc(x1[1], x1[2])

            kp1_loc = sample_loc(loc1, kp1)

            endpoint[eu.KP1] = endpoint[eu.KP1] + kp1_loc

        else:
            x1, x2 = bundle[X1], bundle[X2]
            kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]

            loc1 = self.loc(x1[1], x1[2])
            loc2 = self.loc(x2[1], x2[2])

            kp1_loc = sample_loc(loc1, kp1)
            kp2_loc = sample_loc(loc2, kp2)

            # TODO. What if a point is outside of the range?

            endpoint[eu.KP1] = endpoint[eu.KP1] + kp1_loc
            endpoint[eu.KP2] = endpoint[eu.KP2] + kp2_loc

        return endpoint, bundle


class DescJointBranchWrapper(ModuleWrapper):

    def __init__(self, device, model_config):
        super().__init__(device)
        self.descriptor = DescBranch.from_config(model_config)

        self.deploy_mode = model_config.get(exp.DEPLOY_MODE)
        self.grid_size = model_config[exp.GRID_SIZE]

    def forward(self, engine, batch, endpoint, bundle):
        if self.deploy_mode is not None and self.deploy_mode:
            x1 = bundle[X1]
            kp1 = endpoint[eu.KP1]

            desc1 = self.descriptor(x1[0])

            kp1_desc = sample_descriptors(desc1, kp1, self.grid_size)

            endpoint[eu.KP1_DESC] = kp1_desc

        else:
            x1, x2 = bundle[X1], bundle[X2]
            kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]

            desc1 = self.descriptor(x1[0])
            desc2 = self.descriptor(x2[0])

            kp1_desc = sample_descriptors(desc1, kp1, self.grid_size)
            kp2_desc = sample_descriptors(desc2, kp2, self.grid_size)

            endpoint[eu.DESC1] = desc1
            endpoint[eu.DESC2] = desc2

            endpoint[eu.KP1_DESC] = kp1_desc
            endpoint[eu.KP2_DESC] = kp2_desc

        return endpoint, bundle


# Legacy wrappers


# class DetMSBranchWrapper(ModuleWrapper):
#
#     def __init__(self, det, model_config):
#         super().__init__()
#         self.det = det
#
#         self.nms_kernel_size = model_config[exp.NMS_KERNEL_SIZE]
#         self.top_k = model_config.get(exp.TOP_K)
#         self.score_thresh = model_config.get(exp.SCORE_THRESH)
#
#     def forward(self, engine, batch, endpoint, bundle):
#         x1, x2 = bundle[X1], bundle[X2]
#         image1, image2 = batch.get(du.IMAGE1), batch.get(du.IMAGE2)
#
#         score1, model_info1 = self.det(x1[0], x1[1:])
#         score2, model_info2 = self.det(x2[0], x2[1:])
#
#         kp1 = select_kp(score1, self.top_k, self.nms_kernel_size, self.score_thresh)
#         kp2 = select_kp(score2, self.top_k, self.nms_kernel_size, self.score_thresh)
#
#         endpoint[eu.SCORE1] = score1
#         endpoint[eu.SCORE2] = score2
#
#         endpoint[eu.KP1] = kp1
#         endpoint[eu.KP2] = kp2
#
#         endpoint[eu.MODEL_INFO1] = model_info1
#         endpoint[eu.MODEL_INFO2] = model_info2
#
#         w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask = warp_points(kp1, kp2, image1, image2, batch)
#
#         endpoint[eu.W_KP1] = w_kp1
#         endpoint[eu.W_KP2] = w_kp2
#
#         endpoint[eu.W_VIS_KP1_MASK] = w_vis_kp1_mask
#         endpoint[eu.W_VIS_KP2_MASK] = w_vis_kp2_mask
#
#         return endpoint, bundle
#
#
# class LocMSBranchWrapper(ModuleWrapper):
#
#     def __init__(self, model_config):
#         super().__init__()
#         self.loc = LocOldBranch.from_config(model_config)
#
#     def forward(self, engine, batch, endpoint, bundle):
#         kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
#
#         loc1 = self.loc(endpoint[eu.MODEL_INFO1][mu.LOG_MS_SCORE])
#         loc2 = self.loc(endpoint[eu.MODEL_INFO2][mu.LOG_MS_SCORE])
#
#         kp1_loc = sample_loc(loc1, kp1)
#         kp2_loc = sample_loc(loc2, kp2)
#
#         endpoint[eu.KP1] = endpoint[eu.KP1].float() + kp1_loc
#         endpoint[eu.KP2] = endpoint[eu.KP2].float() + kp2_loc
#
#         return endpoint, bundle
#
#
# class DescMSBranchWrapper(ModuleWrapper):
#
#     def __init__(self, model_config):
#         super().__init__()
#         self.desc = DescOldBranch.from_config(model_config)
#
#         self.grid_size = model_config[exp.GRID_SIZE]
#
#     def forward(self, engine, batch, endpoint, bundle):
#         x1, x2 = bundle[X1], bundle[X2]
#         kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
#
#         desc1 = self.desc(x1[0])
#         desc2 = self.desc(x2[0])
#
#         kp1_desc = sample_descriptors(desc1, kp1, self.grid_size)
#         kp2_desc = sample_descriptors(desc2, kp2, self.grid_size)
#
#         endpoint[eu.DESC1] = desc1
#         endpoint[eu.DESC2] = desc2
#
#         endpoint[eu.KP1_DESC] = kp1_desc
#         endpoint[eu.KP2_DESC] = kp2_desc
#
#         return endpoint, bundle
