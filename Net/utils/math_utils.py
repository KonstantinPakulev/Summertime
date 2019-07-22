import torch


def distance_matrix(t1, t2):
    """
    :param t1: B x C x N
    :param t2: B x C x N
    :return d_matrix: B x N x N
    """

    d_matrix = 2 - 2 * torch.bmm(t1.transpose(1, 2), t2) # [0, 4]
    d_matrix = d_matrix.clamp(min=1e-8, max=4.0)
    d_matrix = torch.sqrt(d_matrix) # [0, 2]

    return d_matrix