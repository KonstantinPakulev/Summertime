import torch

from Net.source.utils.matching_utils import get_mutual_desc_matches, DescriptorDistance


# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(descriptors1, descriptors2):
    # device = descriptors1.device
    # sim = descriptors1 @ descriptors2.t()
    # nn12 = torch.max(sim, dim=1)[1]
    # nn21 = torch.max(sim, dim=0)[1]
    # ids1 = torch.arange(0, sim.shape[0], device=device)
    # mask = ids1 == nn21[nn12]
    # matches = torch.stack([ids1[mask], nn12[mask]]).t()
    # return matches.data.cpu().numpy()

    match_mask, nn_desc_id1 = get_mutual_desc_matches(descriptors1.unsqueeze(0),
                                                      descriptors2.unsqueeze(0),
                                                      DescriptorDistance.INV_COS_SIM, 0.9)

    ids1 = torch.arange(0, descriptors1.shape[0], device=descriptors1.device)

    matches = torch.stack([ids1[match_mask[0]], nn_desc_id1[0][match_mask[0]]]).t()
    return matches.data.cpu().numpy()


