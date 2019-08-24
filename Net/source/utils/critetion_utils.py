import torch
import numpy as np

from Net.source.utils.image_utils import (create_desc_coordinates_grid,
                                          create_desc_center_coordinates_grid,
                                          warp_points)
from Net.source.utils.math_utils import calculate_inv_similarity_matrix, calculate_inv_similarity_vector, \
    calculate_distance_matrix
from Net.source.utils.model_utils import sample_descriptors


def calculate_interpolation_fos(kp1, w_kp1, kp1_desc, desc2, homo12, grid_size, margin, num_neg):
    b, n, c = kp1_desc.size()

    w_kp1_desc = sample_descriptors(desc2, w_kp1, grid_size)

    # Take positive matches
    positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
    positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, num_neg).view(b, n * num_neg)

    # Create neighbour mask
    coo_grid = create_desc_coordinates_grid(desc2, grid_size).view(desc2.size(0), -1, 2).to(desc2.device)

    kp1_coo_dist = calculate_distance_matrix(kp1, coo_grid)
    _, kp1_cell_ids = kp1_coo_dist.topk(k=4, largest=False, dim=-1)

    kp1_cell_ids = kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
    kp1_cells = coo_grid.gather(dim=1, index=kp1_cell_ids)

    kp1_w_cells = warp_points(kp1_cells, homo12)

    kp1_wc_coo_dist = calculate_distance_matrix(kp1_w_cells, coo_grid)
    _, kp1_w_cell_cell_ids = kp1_wc_coo_dist.topk(k=4, largest=False, dim=-1)

    neigh_mask = torch.zeros_like(kp1_wc_coo_dist).to(kp1_wc_coo_dist)
    neigh_mask = neigh_mask.scatter(dim=-1, index=kp1_w_cell_cell_ids, value=1)
    neigh_mask = neigh_mask.view(b, n, 4, -1).sum(dim=2).float()

    # Calculate similarity
    desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
    desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)

    #  Apply neighbour mask and get negatives
    desc_sim = desc_sim + neigh_mask * 5
    neg_sim = desc_sim.topk(k=num_neg, dim=-1, largest=False)[0].view(b, -1)

    fos = (torch.clamp(positive_sim - neg_sim + margin, min=0) ** 2).mean()

    return fos, w_kp1_desc, kp1_cells, kp1_w_cell_cell_ids, coo_grid


def calculate_interpolation_sos(kp1_cells, kp1_w_cell_cell_ids, coo_grid, w_kp1, kp1_desc, w_kp1_desc, sos_neg):
    b, n, c = kp1_desc.size()

    # Create kp1 neighbour mask
    kp1_cells = kp1_cells.view(b, n * 4, 2)
    kp1_cells_dist = calculate_distance_matrix(kp1_cells, kp1_cells)
    kp1_mask = (kp1_cells_dist <= 1e-8).view(b, n, 4, n, 4).sum(-1).sum(2).float()

    # Create w_kp1 neighbour mask
    kp1_w_cell_cell_ids = kp1_w_cell_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
    kp1_w_cell_cells = coo_grid.gather(dim=1, index=kp1_w_cell_cell_ids)

    w_kp1_coo_dist = calculate_distance_matrix(w_kp1, coo_grid)
    _, w_kp1_cell_ids = w_kp1_coo_dist.topk(k=4, largest=False, dim=-1)

    w_kp1_cell_ids = w_kp1_cell_ids.view(b, -1).unsqueeze(-1).repeat(1, 1, 2)
    w_kp1_cells = coo_grid.gather(dim=1, index=w_kp1_cell_ids)

    kp1_wc_w_kp1_dist = calculate_distance_matrix(kp1_w_cell_cells, w_kp1_cells)
    w_kp1_mask = (kp1_wc_w_kp1_dist <= 1e-8).view(b, n, 4, 4, n, 4).sum(-1).sum(3).sum(2).float()

    kp1_sim = calculate_inv_similarity_matrix(kp1_desc, kp1_desc)
    w_kp1_sim = calculate_inv_similarity_matrix(w_kp1_desc, w_kp1_desc)

    kp1_sim = kp1_sim + kp1_mask * 5
    w_kp1_sim = w_kp1_sim + w_kp1_mask * 5

    _, kp1_neg_ids = kp1_sim.topk(k=sos_neg, dim=-1, largest=False)
    _, w_kp1_neg_ids = w_kp1_sim.topk(k=sos_neg, dim=-1, largest=False)

    kp1_neg_ids = kp1_neg_ids.view(b, n * sos_neg).unsqueeze(-1).repeat(1, 1, c)
    w_kp1_neg_ids = w_kp1_neg_ids.view(b, n * sos_neg).unsqueeze(-1).repeat(1, 1, c)

    kp1_neg_desc = kp1_desc.gather(dim=1, index=kp1_neg_ids)
    w_kp1_neg_desc = w_kp1_desc.gather(dim=1, index=w_kp1_neg_ids)

    kp1_desc = kp1_desc.unsqueeze(2).repeat(1, 1, sos_neg, 1).view(b, n * sos_neg, c)
    w_kp1_desc = w_kp1_desc.unsqueeze(2).repeat(1, 1, sos_neg, 1).view(b, n * sos_neg, c)

    sos = calculate_inv_similarity_vector(kp1_desc, kp1_neg_desc) - \
          calculate_inv_similarity_vector(w_kp1_desc, w_kp1_neg_desc)

    sos = (sos ** 2).view(b, n, sos_neg).sum(-1).sqrt().mean()

    return sos


def calculate_radius_fos(w_kp1, kp1_desc, desc2, grid_size, margin, num_neg):
    b, n, c = kp1_desc.size()

    radius = grid_size * np.sqrt(2) + 0.1

    w_kp1_desc = sample_descriptors(desc2, w_kp1, grid_size)

    # Take positive matches
    positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
    positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, num_neg).view(b, n * num_neg)

    # Create neighbour mask
    coo_grid = create_desc_coordinates_grid(desc2, grid_size).view(desc2.size(0), -1, 2).to(desc2.device)
    neigh_mask = (calculate_distance_matrix(w_kp1, coo_grid) <= radius).float()

    # Calculate similarity
    desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
    desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)

    #  Apply neighbour mask and get negatives
    desc_sim = desc_sim + neigh_mask * 5
    neg_sim = desc_sim.topk(k=num_neg, dim=-1, largest=False)[0].view(b, -1)

    fos = (torch.clamp(positive_sim - neg_sim + margin, min=0) ** 2).mean()

    return fos


def calculate_center_radius_fos(w_kp1, kp1_desc, desc2, grid_size, margin, num_neg):
    b, n, c = kp1_desc.size()

    radius = grid_size * np.sqrt(2) + 0.1

    w_kp1_desc = sample_descriptors(desc2, w_kp1, grid_size)

    # Take positive matches
    positive_sim = calculate_inv_similarity_vector(kp1_desc, w_kp1_desc)
    positive_sim = positive_sim.view(b, n, 1).repeat(1, 1, num_neg).view(b, n * num_neg)

    # Create neighbour mask
    coo_grid = create_desc_center_coordinates_grid(desc2, grid_size).view(desc2.size(0), -1, 2).to(desc2.device)
    neigh_mask = (calculate_distance_matrix(w_kp1, coo_grid) <= radius).float()

    # Calculate similarity
    desc2 = desc2.permute((0, 2, 3, 1)).view(b, -1, c)
    desc_sim = calculate_inv_similarity_matrix(kp1_desc, desc2)

    #  Apply neighbour mask and get negatives
    desc_sim = desc_sim + neigh_mask * 5
    neg_sim = desc_sim.topk(k=num_neg, dim=-1, largest=False)[0].view(b, -1)

    fos = (torch.clamp(positive_sim - neg_sim + margin, min=0) ** 2).mean()

    return fos


def calculate_radius_sos(kp1, w_kp1, kp1_desc, w_kp1_desc, grid_size, sos_neg):
    b, n, c = kp1_desc.size()

    radius = grid_size * np.sqrt(2) + 0.1

    kp1_neigh = (calculate_distance_matrix(kp1, kp1) <= radius).float()
    w_kp1_negh = (calculate_distance_matrix(w_kp1, w_kp1) <= radius).float()

    kp1_sim = calculate_inv_similarity_matrix(kp1_desc, kp1_desc)
    w_kp1_sim = calculate_inv_similarity_matrix(w_kp1_desc, w_kp1_desc)

    kp1_sim = kp1_sim + kp1_neigh * 5
    w_kp1_sim = w_kp1_sim + w_kp1_negh * 5

    _, kp1_neg_ids = kp1_sim.topk(k=sos_neg, dim=-1, largest=False)
    _, w_kp1_neg_ids = w_kp1_sim.topk(k=sos_neg, dim=-1, largest=False)

    kp1_neg_ids = kp1_neg_ids.view(b, n * sos_neg).unsqueeze(-1).repeat(1, 1, c)
    w_kp1_neg_ids = w_kp1_neg_ids.view(b, n * sos_neg).unsqueeze(-1).repeat(1, 1, c)

    kp1_neg_desc = kp1_desc.gather(dim=1, index=kp1_neg_ids)
    w_kp1_neg_desc = w_kp1_desc.gather(dim=1, index=w_kp1_neg_ids)

    kp1_desc = kp1_desc.unsqueeze(2).repeat(1, 1, sos_neg, 1).view(b, n * sos_neg, c)
    w_kp1_desc = w_kp1_desc.unsqueeze(2).repeat(1, 1, sos_neg, 1).view(b, n * sos_neg, c)

    sos = calculate_inv_similarity_vector(kp1_desc, kp1_neg_desc) - \
          calculate_inv_similarity_vector(w_kp1_desc, w_kp1_neg_desc)

    sos = (sos ** 2).view(b, n, sos_neg).sum(-1).sqrt().mean()

    return sos
