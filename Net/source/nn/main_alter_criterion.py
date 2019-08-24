import torch
import torch.nn as nn

from Net.source.utils.critetion_utils import calculate_interpolation_fos, calculate_radius_fos, \
    calculate_center_radius_fos, calculate_radius_sos


class HardQuadTripletRadiusSOSRLoss(nn.Module):

    def __init__(self, grid_size, margin, num_neg, sos_neg, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin
        self.num_neg = num_neg

        self.sos_neg = sos_neg

        self.loss_lambda = loss_lambda

    def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
        """
        :param kp1 B x N x 2
        :param w_kp1: B x N x 2
        :param kp1_desc: B x N x C
        :param desc2: B x C x H x W
        :param homo12: B x 3 x 3
        :return: float
        """
        fos, w_kp1_desc = calculate_interpolation_fos(kp1, w_kp1, kp1_desc, desc2, homo12, self.grid_size, self.margin,
                                          self.num_neg)[:2]
        sos = calculate_radius_sos(kp1, w_kp1, kp1_desc, w_kp1_desc, self.grid_size, self.sos_neg)

        loss = (fos + sos) * self.loss_lambda

        return loss


class HardQuadTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, num_neg, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin
        self.num_neg = num_neg

        self.loss_lambda = loss_lambda

    def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
        """
        :param kp1 B x N x 2
        :param w_kp1: B x N x 2
        :param kp1_desc: B x N x C
        :param desc2: B x C x H x W
        :param homo12: B x 3 x 3
        :return: float
        """
        fos = calculate_interpolation_fos(kp1, w_kp1, kp1_desc, desc2, homo12, self.grid_size, self.margin,
                                          self.num_neg)[0]

        loss = fos * self.loss_lambda

        return loss


class HardQuadRadiusTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, num_neg, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin
        self.num_neg = num_neg

        self.loss_lambda = loss_lambda

    def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
        """
        :param kp1 B x N x 2
        :param w_kp1: B x N x 2
        :param kp1_desc: B x N x C
        :param desc2: B x C x H x W
        :param homo12: B x 3 x 3
        :return: float
        """
        fos = calculate_radius_fos(w_kp1, kp1_desc, desc2, self.grid_size, self.margin, self.num_neg)

        loss = fos * self.loss_lambda

        return loss


class HardQuadCenterRadiusTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, num_neg, loss_lambda):
        super().__init__()

        self.grid_size = grid_size

        self.margin = margin
        self.num_neg = num_neg

        self.loss_lambda = loss_lambda

    def forward(self, kp1, w_kp1, kp1_desc, desc2, homo12):
        """
        :param kp1 B x N x 2
        :param w_kp1: B x N x 2
        :param kp1_desc: B x N x C
        :param desc2: B x C x H x W
        :param homo12: B x 3 x 3
        :return: float
        """
        fos = calculate_center_radius_fos(w_kp1, kp1_desc, desc2, self.grid_size, self.margin, self.num_neg)

        loss = fos * self.loss_lambda

        return loss
