import torch
import torch.nn as nn

from ST_Net.model.st_det_vgg import STDetVGGModule
from ST_Net.utils.image_utils import warp, filter_border, top_k_map
from ST_Net.utils.model_utils import make_vgg_backbone


class STNetVGGModule(nn.Module):
    def __init__(self, det, des):
        super(STNetVGGModule, self).__init__()

        self.backbone = make_vgg_backbone()
        self.det = det
        self.des = des

    def forward(self, batch):
        left_image, homo_r2l, right_image, homo_l2r = batch

        left_features = self.backbone(left_image)
        right_features = self.backbone(right_image)

        left_raw_score = self.det(left_features)
        right_raw_score = self.det(right_features)

        left_raw_des, left_des = self.des(left_features)
        right_raw_des, right_des = self.des(right_features)

        left_gt, left_top_k_mask, left_top_k_value, left_visible_mask = self.get_gt_score(right_raw_score, homo_r2l)
        right_gt, right_top_k_mask, right_top_k_value, right_visible_mask = self.get_gt_score(left_raw_score, homo_l2r)

        c_mask = self.get_des_correspondence_mask(right_raw_des, homo_r2l)

        left_score = self.det.process(left_raw_score)[0]
        right_score = self.det.process(right_raw_score)[0]

        endpoint = {"left_score": left_score,
                    "left_gt": left_gt,
                    "left_visible_mask": left_visible_mask,
                    "left_raw_des": left_raw_des,
                    "c_mask": c_mask,

                    "right_score": right_score,
                    "right_gt": right_gt,
                    "right_visible_mask": right_visible_mask,
                    "right_raw_des": right_raw_des
                    }

        return endpoint

    def get_gt_score(self, right_raw_score, homography):
        right_raw_score = filter_border(right_raw_score)

        # warp right_raw_score to left_raw_score and calculate visible_mask
        left_raw_score = warp(right_raw_score, homography)
        left_visible_mask = warp(
            right_raw_score.new_full(right_raw_score.size(), fill_value=1, requires_grad=True),
            homography
        )

        left_gt, left_top_k_mask, left_top_k_value = self.det.process(left_raw_score)

        return left_gt, left_top_k_mask, left_top_k_value, left_visible_mask

    @staticmethod
    def get_des_correspondence_mask(right_raw_des, homography):
        n, c, h, w = right_raw_des.size()

        out_h, out_w = h, w
        gy, gx = torch.meshgrid([torch.arange(out_h), torch.arange(out_w)])
        gx, gy = gx.float().unsqueeze(-1), gy.float().unsqueeze(-1)

        ones = gy.new_full(gy.size(), fill_value=1)
        grid = torch.cat((gx, gy), -1) * 8 + 4
        grid = torch.cat((grid, ones), -1)  # (H, W, 3)
        grid = grid.unsqueeze(0)  # (1, H, W, 3)
        grid = grid.repeat(n, 1, 1, 1)  # (B, H, W, 3)

        initial_grid = grid[:, :, :, :2].to(homography.device)

        grid = grid.view(grid.size(0), -1, grid.size(-1))  # (B, H*W, 3)
        grid = grid.permute(0, 2, 1)  # (B, 3, H*W)
        grid = grid.type_as(homography).to(homography.device)

        # (B, 3, 3) matmul (B, 3, H*W) => (B, 3, H*W)
        grid_w = torch.matmul(homography, grid)
        grid_w = grid_w.permute(0, 2, 1)  # (B, H*W, 3)
        grid_w = grid_w.div(grid_w[:, :, 2].unsqueeze(-1) + 1e-8)  # (B, H*W, 3)
        grid_w = grid_w.view(n, out_h, out_w, -1)[:, :, :, :2]  # (B, H, W, 2)

        initial_grid = initial_grid.reshape([n, h, w, 1, 1, 2])
        grid_w = grid_w.reshape([n, 1, 1, h, w, 2])
        cell_distance = torch.norm(initial_grid - grid_w, dim=-1)

        s = torch.where(cell_distance <= 8 - 0.5, torch.ones_like(cell_distance, dtype=torch.float),
                        torch.zeros_like(cell_distance, dtype=torch.float)).to(homography.device)

        return s

    def criterion(self, endpoint):
        left_score = endpoint['left_score']
        left_gt = endpoint['left_gt']
        left_visible_mask = endpoint['left_visible_mask']
        left_raw_des = endpoint['left_raw_des']

        right_score = endpoint['right_score']
        right_gt = endpoint['right_gt']
        right_visible_mask = endpoint['right_visible_mask']
        right_raw_des = endpoint['right_raw_des']

        c_mask = endpoint["c_mask"]

        left_score_loss = self.det.loss(left_score, left_gt, left_visible_mask)
        right_score_loss = self.det.loss(right_score, right_gt, right_visible_mask)
        det_loss = (left_score_loss + right_score_loss) / 2.0

        des_loss = self.des.loss(left_raw_des, right_raw_des, c_mask)

        plt_scalar = {'det_loss': det_loss, 'des_loss': des_loss}

        plt = {'scalar': plt_scalar}

        return plt, det_loss.mean(), des_loss

    def inference(self, im_data, show_inference_time=False):
        # start_det = torch.cuda.Event(enable_timing=True)
        # end_det = torch.cuda.Event(enable_timing=True)

        # start_det.record()

        features = self.backbone(im_data)
        raw_score_map = self.det(features)

        im_score = self.det.process(raw_score_map)[0]
        im_topk = top_k_map(im_score, 512)
        kpts = im_topk.nonzero()  # (B*topk, 4)

        raw_desc, desc = self.des(features)

        selected_desc = desc[0][:, kpts[:, 2], kpts[:, 3]].permute(1, 0)

        # end_det.record()
        #
        # torch.cuda.synchronize()

        # if show_inference_time:
        #     print(f"Detector inference time is: {start_det.elapsed_time(end_det):.2f}")


        # print(desc.shape)
        # print(kpts.shape)

        return kpts, selected_desc

    def detectAndCompute(self, im_path, device, output_size):
        """
        detect keypoints and compute its descriptor
        :param im_path: image path
        :param device: cuda or cpu
        :param output_size: rescale size
        :return: kp (#keypoints, 4) des (#keypoints, 128)
        """
        import numpy as np
        from skimage import io, color
        from ST_Net.utils.image_utils import im_rescale

        img = io.imread(im_path)
        img = np.expand_dims(color.rgb2gray(img), -1)

        # Rescale
        # output_size = (240, 320)
        img, _, _, sw, sh = im_rescale(img, output_size)

        # to tensor
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = torch.from_numpy(img.transpose((2, 0, 1)))[None, :].to(
            device, dtype=torch.float
        )

        # inference
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        kp, desc = self.inference(img, show_inference_time=True)
        # end.record()

        # torch.cuda.synchronize()
        # print(f"Total inference time is: {start.elapsed_time(end):.2f}")

        return kp, desc, img
