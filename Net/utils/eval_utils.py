import torch


def nearest_neighbor_match_score(s, dot_dest, r_mask):
    """
    :param s: N x H x W x H x W. Matrix of correspondences
    :param dot_dest: N x H x W x H x W. Matrix of distances
    :param r_mask: N x 1 x H x W
    """
    n, h, w, _, _ = s.size()
    flat = h * w

    s = s.view(n, flat, flat)
    dot_dest = dot_dest.view(n, flat, flat)
    r_mask = r_mask.view(n, 1, flat)

    ids0 = torch.ones((n, flat), dtype=torch.long) * torch.arange(0, n).view(n, 1)
    ids1 = torch.arange(0, flat).view(1, flat).repeat((n, 1))
    # A pair of normalized descriptors will have the maximum value of scalar product if they are the closets
    _, ids2 = dot_dest.max(dim=-1)

    n_flat = n * flat

    ids0 = ids0.view(n_flat, 1).to(s.device)
    ids1 = ids1.view(n_flat, 1).to(s.device)
    ids2 = ids2.view(n_flat, 1)
    ids = torch.cat((ids0, ids1, ids2), dim=-1)

    s = s * r_mask

    # Consider only unique matches between descriptors of total number of possible matches
    matches = s[ids[:, 0], ids[:, 1], ids[:, 2]]
    cm_ids = matches.nonzero()
    correct_matches = torch.zeros((flat,)).to(s.device)
    correct_matches[ids[cm_ids, 2]] = 1

    correct_matches = correct_matches.sum()
    total = r_mask.sum()

    # TODO. Try to consider some threshold or n closest neighbours. #LATER

    return correct_matches / total

