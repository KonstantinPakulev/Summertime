import torch.nn as nn

from Net._legacy.source.utils.critetion_utils import calculate_interpolation_fos, calculate_radius_fos, \
    calculate_center_radius_fos, calculate_radius_sos, calculate_interpolation_sos


class HardQuadTripletSOSRLoss(nn.Module):

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
        # Calculate FOS
        fos, w_kp1_desc, kp1_cells, kp1_w_cell_cell_ids, coo_grid = \
            calculate_interpolation_fos(kp1, w_kp1, kp1_desc, desc2, homo12, self.grid_size, self.margin, self.num_neg)

        # Calculate SOS
        sos = calculate_interpolation_sos(kp1_cells, kp1_w_cell_cell_ids, coo_grid, w_kp1, kp1_desc, w_kp1_desc, self.sos_neg)

        loss = (fos + sos) * self.loss_lambda

        return loss


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


class DenseQTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin

        self.loss_lambda = loss_lambda

    def forward(self, desc1, desc2, homo12, w_vis_mask1, score2):
        b, c, hc, wc = desc1.size()

        coo_grid1 = create_desc_coordinates_grid(desc1, self.grid_size, False).to(desc1.device)
        w_coo_grid1 = warp_coordinates_grid(coo_grid1, homo12).view(b, -1, 2)
        w_coo_grid1 = w_coo_grid1[:, :, [1, 0]]

        w_desc1 = sample_descriptors(desc2, w_coo_grid1, self.grid_size)

        # Create neigh mask
        coo_grid2 = create_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
        coo_dist = calculate_distance_matrix(w_coo_grid1, coo_grid2)
        neigh_mask = (coo_dist <= self.grid_size).float()

        # Prepare visibility mask
        w_vis_mask1 = space_to_depth(w_vis_mask1, self.grid_size)
        w_vis_mask1 = w_vis_mask1.prod(dim=1).unsqueeze(1).view(b, 1, -1)

        # Get positive and negative matches
        desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(b, c, -1).permute((0, 2, 1))

        desc_sim = calculate_inv_similarity_matrix(desc1, desc2)
        desc_sim = desc_sim + neigh_mask * 5
        desc_sim = desc_sim + (1 - w_vis_mask1.float()) * 5

        wv_match_mask1 = get_visible_keypoints_mask(score2, w_coo_grid1).float()

        pos_sim = calculate_inv_similarity_vector(desc1, w_desc1)
        neg_sim = desc_sim.min(dim=-1)[0]

        loss = torch.clamp(pos_sim - neg_sim + self.margin, min=0) ** 2 * wv_match_mask1
        loss = loss.sum() / wv_match_mask1.sum() * self.loss_lambda

        return loss


class DenseInterTripletLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin

        self.loss_lambda = loss_lambda


    def forward(self, desc1, desc2, homo12, w_vis_mask1, score2):
        b, c, hc, wc = desc1.size()
        flat = hc * wc

        coo_grid1 = create_desc_coordinates_grid(desc1, self.grid_size, False).to(desc1.device)
        w_coo_grid1 = warp_coordinates_grid(coo_grid1, homo12).view(b, -1, 2)
        w_coo_grid1 = w_coo_grid1[:, :, [1, 0]]

        w_desc1 = sample_descriptors(desc2, w_coo_grid1, self.grid_size)

        # Create neigh mask
        coo_grid2 = create_center_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
        coo_dist = calculate_distance_matrix(w_coo_grid1, coo_grid2)
        _, ul = coo_dist.min(dim=-1)

        ul = ul.unsqueeze(-1)

        ur = ul + 1
        ur = torch.where(ur >= flat, ul, ur)

        ll = ul + wc
        ll = torch.where(ll >= flat, ul, ll)

        lr = ll + 1
        lr = torch.where(lr >= flat, ul, lr)

        neigh_mask_ids = torch.cat([ul, ur, ll, lr], dim=-1)

        neigh_mask = torch.zeros_like(coo_dist).to(coo_dist.device)
        neigh_mask = neigh_mask.scatter(dim=-1, index=neigh_mask_ids, value=1)

        # Prepare visibility mask
        w_vis_mask1 = space_to_depth(w_vis_mask1, self.grid_size)
        w_vis_mask1 = w_vis_mask1.prod(dim=1).unsqueeze(1).view(b, 1, -1)

        # Get positive and negative matches
        desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(b, c, -1).permute((0, 2, 1))

        desc_sim = calculate_inv_similarity_matrix(desc1, desc2)
        desc_sim = desc_sim + neigh_mask * 5
        desc_sim = desc_sim + (1 - w_vis_mask1.float()) * 5

        wv_match_mask1 = get_visible_keypoints_mask(score2, w_coo_grid1).float()

        pos_sim = calculate_inv_similarity_vector(desc1, w_desc1)
        neg_sim = desc_sim.min(dim=-1)[0]

        loss = torch.clamp(pos_sim - neg_sim + self.margin, min=0) * wv_match_mask1
        loss = loss.sum() / wv_match_mask1.sum() * self.loss_lambda

        return loss


class DenseInterQTripletSOSRLoss(nn.Module):

    def __init__(self, grid_size, margin, loss_lambda):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin

        self.loss_lambda = loss_lambda

    def forward(self, desc1, desc2, homo12, w_vis_mask1, score2):
        b, c, hc, wc = desc1.size()
        flat = hc * wc

        coo_grid1 = create_desc_coordinates_grid(desc1, self.grid_size, False).to(desc1.device)
        w_coo_grid1 = warp_coordinates_grid(coo_grid1, homo12).view(b, -1, 2)
        w_coo_grid1 = w_coo_grid1[:, :, [1, 0]]

        w_desc1 = sample_descriptors(desc2, w_coo_grid1, self.grid_size)

        # Create neigh mask
        coo_grid2 = create_center_desc_coordinates_grid(desc2, self.grid_size).view(b, -1, 2).to(desc2.device)
        coo_dist = calculate_distance_matrix(w_coo_grid1, coo_grid2)
        _, ul = coo_dist.min(dim=-1)

        ul = ul.unsqueeze(-1)

        ur = ul + 1
        ur = torch.where(ur >= flat, ul, ur)

        ll = ul + wc
        ll = torch.where(ll >= flat, ul, ll)

        lr = ll + 1
        lr = torch.where(lr >= flat, ul, lr)

        neigh_mask_ids = torch.cat([ul, ur, ll, lr], dim=-1)

        neigh_mask = torch.zeros_like(coo_dist).to(coo_dist.device)
        neigh_mask = neigh_mask.scatter(dim=-1, index=neigh_mask_ids, value=1)

        # Prepare visibility mask
        w_vis_mask1 = space_to_depth(w_vis_mask1, self.grid_size)
        w_vis_mask1 = w_vis_mask1.prod(dim=1).unsqueeze(1).view(b, 1, -1)

        # Get positive and negative matches
        desc1 = desc1.view(b, c, -1).permute((0, 2, 1))
        desc2 = desc2.view(b, c, -1).permute((0, 2, 1))

        desc_sim = calculate_inv_similarity_matrix(desc1, desc2)
        desc_sim = desc_sim + neigh_mask * 5
        desc_sim = desc_sim + (1 - w_vis_mask1.float()) * 5

        wv_match_mask1 = get_visible_keypoints_mask(score2, w_coo_grid1).float()

        pos_sim = calculate_inv_similarity_vector(desc1, w_desc1)
        neg_sim = desc_sim.min(dim=-1)[0]

        fos = torch.clamp(pos_sim - neg_sim + self.margin, min=0) ** 2 * wv_match_mask1
        fos = fos.sum() / wv_match_mask1.sum()

        sos_neg = 4

        cells1_mask = torch.eye(flat).repeat(b, 1, 1).to(desc1.device)

        neigh_mask_ids = neigh_mask_ids.view(b, -1)

        w_cell1_dist = calculate_difference_matrix(neigh_mask_ids, neigh_mask_ids)
        w_cell1_mask = w_cell1_dist.eq(0).view(b, flat, 4, flat, 4).sum(-1).sum(-2).float()

        cells1_sim = calculate_inv_similarity_matrix(desc1, desc1)
        cells1_sim = cells1_sim + cells1_mask * 5

        w_cells1_sim = calculate_inv_similarity_matrix(w_desc1, w_desc1)
        w_cells1_sim = w_cells1_sim + w_cell1_mask * 5

        _, cells1_neg_ids = cells1_sim.topk(k=sos_neg, dim=-1, largest=False)
        _, w_cells1_neg_ids = w_cells1_sim.topk(k=sos_neg, dim=-1, largest=False)

        cells1_neg_ids = cells1_neg_ids.view(b, flat * sos_neg).unsqueeze(-1).repeat(1, 1, c)
        w_cells1_neg_ids = w_cells1_neg_ids.view(b, flat * sos_neg).unsqueeze(-1).repeat(1, 1, c)

        cells1_neg_desc = desc1.gather(dim=1, index=cells1_neg_ids)
        w_cells1_neg_desc = w_desc1.gather(dim=1, index=w_cells1_neg_ids)

        desc1 = desc1.unsqueeze(2).repeat(1, 1, sos_neg, 1).view(b, flat * sos_neg, c)
        w_desc1 = w_desc1.unsqueeze(2).repeat(1, 1, sos_neg, 1).view(b, flat * sos_neg, c)

        sos = calculate_inv_similarity_vector(desc1, cells1_neg_desc) - calculate_inv_similarity_vector(w_desc1, w_cells1_neg_desc)
        sos = (sos ** 2).view(b, flat, sos_neg).sum(-1).sqrt().mean()

        loss = fos + sos
        loss = loss * self.loss_lambda

        return loss
