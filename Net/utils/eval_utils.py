import torch

from Net.utils.math_utils import distance_matrix


def nearest_neighbor_match_score(t1, t2):
    """
    :param t1: N x C x H x W
    :param t2: N x C x H x W
    """

    n, c, h, w = t1.size()

    d_matrix = distance_matrix(t1.view(n, c, -1),
                               t2.view(n, c, -1))

    _, ids = d_matrix.min(dim=-1)

    # TODO. For batches
    correct_matches = torch.arange(0, ids.shape[1]).eq(ids).sum().float()
    # TODO. Replace total with visible
    total = h * w

    return correct_matches / total
