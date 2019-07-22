import torch

from Net.utils.math_utils import distance_matrix


def nearest_neighbor_match_score(t1, t2, mask):
    """
    :param t1: N x C x H x W
    :param t2: N x C x H x W
    :param mask: N x 1 x H x W
    """

    n, c, h, w = t1.size()

    d_matrix = distance_matrix(t1.view(n, c, -1),
                               t2.view(n, c, -1))

    _, ids = d_matrix.min(dim=-1)

    matches = torch.arange(0, ids.shape[1]).type_as(ids).to(ids.device).eq(ids)  # N x H * W
    correct_matches = (matches.type_as(mask) * mask.view(n, -1)).sum()
    total = mask.sum()

    return correct_matches / total
