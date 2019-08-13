import torch


def calculate_inv_similarity_matrix(t1, t2):
    """
    :param t1: B x N1 x C
    :param t2: B x N2 x C
    """
    sim = 2 - 2 * torch.bmm(t1, t2.permute(0, 2, 1))
    sim = sim.clamp(min=1e-8, max=4.0)
    sim = torch.sqrt(sim)
    return sim


def calculate_inv_similarity_vector(t1, t2):
    """
    :param t1: B x N1 x C
    :param t2: B x N2 x C
    """
    sim = 2 - 2 * torch.sum(t1 * t2, dim=-1)
    sim = sim.clamp(min=1e-8, max=4.0)
    sim = torch.sqrt(sim)
    return sim


def calculate_distance_matrix(t1, t2):
    """
    :param t1: B x N1 x 2
    :param t2: B x N2 x 2
    """
    t1 = t1.unsqueeze(2).float()
    t2 = t2.unsqueeze(1).float()

    dist = torch.norm(t1 - t2, p=2, dim=-1)

    return dist

