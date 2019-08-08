import torch


def calculate_similarity_matrix(anchor, positive):
    """
    :param anchor: N x C
    :param positive: N x C
    """
    sim = 2 - 2 * torch.mm(anchor, positive.t())
    sim = sim.clamp(min=1e-8, max=4.0)
    sim = torch.sqrt(sim)
    return sim


def calculate_similarity_vector(anchor, positive):
    sim = 2 - 2 * torch.sum(anchor * positive, dim=-1)
    sim = sim.clamp(min=1e-8, max=4.0)
    sim = torch.sqrt(sim)
    return sim
