import torch

from Net.utils.math_utils import distance_matrix
from Net.utils.eval_utils import nearest_neighbor_match_score


des1 = torch.ones((1, 16, 30, 40))
des1 /= torch.norm(des1, p=2, dim=1).unsqueeze(1)

des2 = torch.ones((1, 30, 40, 16))
des2 /= torch.norm(des2, p=2, dim=1).unsqueeze(1)

print(nearest_neighbor_match_score(des1, des2))