import cv2
import numpy as np

import torch

from torch.nn.functional import normalize, grid_sample

F = 'F'
E = 'E'
E_param = 'param_E'
Rt = 'Rt'


"""
PyTorch distance measurement functions
"""


def smooth_inv_cos_sim(cos_sim):
    """
    :type cos_sim: B x N or B x N1 x N2, cosine similarity [-1, 1], :type torch.tensor, float
    :return B x N, smooth inverted cosine similarity [0, 2], :type torch.tensor, float
    """
    inv_cos_sim = (2 - 2 * cos_sim).clamp(min=1e-8, max=4.0)
    return torch.sqrt(inv_cos_sim)


def smooth_inv_cos_sim_mat(tensor1, tensor2):
    """
    :param tensor1: B x N1 x C, normalized vector, :type torch.tensor, float
    :param tensor2: B x N2 x C, normalized vector, :type torch.tensor, float
    :return inv_cos_sim: B x N1 x N2, :type torch.tensor, float
    """
    cos_sim = torch.bmm(tensor1, tensor2.permute(0, 2, 1))
    return smooth_inv_cos_sim(cos_sim)


def smooth_inv_cos_sim_vec(tensor1, tensor2):
    """
    :param tensor1: B x N x C, normalized vector, :type torch.tensor, float
    :param tensor2: B x N x C, normalized vector, :type torch.tensor, float
    :return inv_cos_sim: B x N, :type torch.tensor, float
    """
    cos_sim = torch.sum(tensor1 * tensor2, dim=-1)
    return smooth_inv_cos_sim(cos_sim)


def inv_cos_sim_mat(tensor1, tensor2):
    """
    :param tensor1: B x N1 x C, normalized vector, :type torch.tensor, float
    :param tensor2: B x N2 x C, normalized vector, :type torch.tensor, float
    :return inv_cos_sim: B x N1 x N2, :type torch.tensor, float
    """
    cos_sim = torch.bmm(tensor1, tensor2.permute(0, 2, 1))
    return 1 - cos_sim


def inv_cos_sim_vec(tensor1, tensor2):
    """
    :param tensor1: B x N x C, normalized vector, :type torch.tensor, float
    :param tensor2: B x N x C, normalized vector, :type torch.tensor, float
    :return inv_cos_sim: B x N, :type torch.tensor, float
    """
    cos_sim = torch.sum(tensor1 * tensor2, dim=-1)
    return 1 - cos_sim


def calculate_distance_vec(tensor1, tensor2):
    """
    :param tensor1: B x N x 2, :type torch.tensor, float
    :param tensor2: B x N x 2, :type torch.tensor, float
    :return dist: B x N, :type torch.tensor, float
    """
    dist = torch.norm(tensor1 - tensor2, p=2, dim=-1)

    return dist


def calculate_distance_mat(tensor1, tensor2):
    """
    :param tensor1: B x N1 x 2, :type torch.tensor, float
    :param tensor2: B x N2 x 2, :type torch.tensor, float
    :return dist: B x N1 x N2, :type torch.tensor, float
    """
    tensor1 = tensor1.unsqueeze(2).float()
    tensor2 = tensor2.unsqueeze(1).float()

    dist = torch.norm(tensor1 - tensor2, p=2, dim=-1)

    return dist


"""
PyTorch angle measurement functions
"""


def angle_mat(R1, R2):
    """
    :param R1: B x 3 x 3, :type torch.float
    :param R2: B x 3 x 3, :type torch.float
    :return: B, angle in degrees, :type torch.float
    """
    R_d = R1.transpose(1, 2) @ R2

    angles = torch.zeros(R1.shape[0]).to(R1.device)
    for i, i_R_d in enumerate(R_d):
        c = (torch.trace(i_R_d) - 1) / 2
        angles[i] = rad2deg(torch.acos(c.clamp(min=-1.0, max=1.0)))

    return angles


def angle_vec(tensor1, tensor2):
    """
    :param tensor1: B x N x C, normalized vector, :type torch.tensor, float
    :param tensor2: B x N x C, normalized vector, :type torch.tensor, float
    :return: B, angle in degrees, :type torch.float
    """
    return rad2deg(torch.acos((tensor1 * tensor2).sum(dim=-1).clamp(min=-1.0, max=1.0)))


"""
PyTorch affine transformations functions
"""


def to_homogeneous(t, dim=-1):
    """
    :param t: Shape B x N x 2 or B x H x W x 3, :type torch.tensor, float
    :param dim: dimension along which to concatenate
    """
    if dim == -1:
        index = len(t.shape) - 1
    else:
        index = dim

    shape = t.shape[:index] + t.shape[index + 1:]
    ones = torch.ones(shape).unsqueeze(dim).float().to(t.device)
    t = torch.cat((t, ones), dim=dim)

    return t


def to_cartesian(t, dim=-1):
    """
    :param t: Shape B x N x 3 or B x H x W x 4, :type torch.tensor, float
    :param dim: dimension along which to normalize
    """
    index = torch.tensor([t.shape[dim] - 1]).to(t.device)
    t = t / torch.index_select(t, dim=dim, index=index).clamp(min=1e-8)

    index = torch.arange(t.shape[dim] - 1).to(t.device)
    t = torch.index_select(t, dim=dim, index=index)

    return t


def revert_data_transform(kp, shift_scale):
    """
    :param kp: B x N x 2
    :param shift_scale: B x 4
    """
    # Convert keypoints from y, x orientation to x, y
    kp = kp[..., [1, 0]].float()

    # Scale and shift image to its original size
    kp = kp / shift_scale[:, None, [3, 2]] + shift_scale[:, None, [1, 0]]

    return kp


def compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2, type=F):
    T12 = extrinsics2 @ extrinsics1.inverse()

    R = T12[:, :3, :3]
    t = T12[:, :3, 3]

    if type == Rt:
        return R, t

    elif type == E_param:
        t = normalize(t, dim=-1)

        b = intrinsics1.shape[0]
        _E_param = torch.zeros(b, 5).to(intrinsics1.device)

        for i in range(b):
            i_E_param = parametrize_pose(R[i].cpu().numpy(), t[i].cpu().numpy())

            _E_param[i] = torch.tensor(i_E_param).to(intrinsics1.device)

        return _E_param

    else:
        E = vec_to_cross(t) @ R

        if type == F:
            return intrinsics2.inverse().transpose(1, 2) @ E @ intrinsics1.inverse()
        else:
            return E


def epipolar_distance(r_kp1, nn_r_kp2, F):
    """
    :param r_kp1: B x N x 2
    :param nn_r_kp2: B x N x 2
    :param F: B x 3 x 3
    :return: B x N
    """
    r_kp1_h = to_homogeneous(r_kp1)
    nn_r_kp2_h = to_homogeneous(nn_r_kp2)

    line2 = r_kp1_h @ F.transpose(1, 2)
    line2 = line2 / line2[..., :2].norm(dim=-1).unsqueeze(-1).clamp(min=1e-16)

    ep_dist = (nn_r_kp2_h * line2).sum(dim=-1).abs()

    return ep_dist


def get_gt_rel_pose(extrinsic1, extrinsic2):
    """
    :param extrinsic1: B x 4 x 4, :type torch.float
    :param extrinsic2: B x 4 x 4, :type torch.float
    :return: (B x 3 x 4) :type torch.float
    """
    T21 = extrinsic1 @ extrinsic2.inverse()

    gt_rel_pose = torch.zeros(extrinsic1.shape[0], 3, 4).to(extrinsic1.device)
    gt_rel_pose[:, :3, :3] = T21[:, :3, :3]
    gt_rel_pose[:, :3, 3] = normalize(T21[:, :3, 3], dim=-1)

    return gt_rel_pose


def change_intrinsics(kp, intrinsics2, intrinsics1):
    """
    :param kp: B x N x 2, :type torch.tensor, float
    :param intrinsics2: B x 3 x 3, initial parameters to set :type torch.tensor, float
    :param intrinsics1: B x 3 x 3, final intrinsic parameters :type torch.tensor, float
    """
    kp_h = to_homogeneous(kp)
    kp_h = kp_h @ torch.inverse(intrinsics2).transpose(1, 2) @ intrinsics1.transpose(1, 2)
    return to_cartesian(kp_h)


def parametrize_pose(R, t):
    R_param, jac = cv2.Rodrigues(R)

    vec = np.asarray([0, 0, 1])
    t_param = rotate_a_b_axis_angle(vec, t)

    E_param = np.concatenate([R_param.reshape(-1), t_param[0:2]], axis=0)

    return E_param


"""
PyTorch grid manipulation functions
"""


def create_coord_grid(shape, center=True, scale_factor=1.0):
    """
    :param shape: (b, c, h, w) :type tuple
    :param scale_factor: float
    :param center: bool
    :return B x H x W x 2; x, y orientation of coordinates located in center of pixels :type torch.tensor, float
    """
    b, _, h, w = shape

    grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])

    grid_x = grid_x.float().unsqueeze(-1)
    grid_y = grid_y.float().unsqueeze(-1)
    grid = torch.cat([grid_x, grid_y], dim=-1)  # H x W x 2

    # Each coordinate represents the location of the center of a pixel
    if center:
        grid += 0.5

    grid *= scale_factor

    return grid.unsqueeze(0).repeat(b, 1, 1, 1)


def sample_grid(data, grid):
    """
    :param data: N x C x H_in x W_in
    :param grid: N x H_out x W_out x 2; Grid have have (x,y) coordinates orientation
    :return N x C x H_out x W_out
    """
    h, w = data.shape[2:]

    # Make a copy to avoid in-place modification
    norm_grid = grid.clone()
    norm_grid[:, :, :, 0] = norm_grid[:, :, :, 0] / w * 2 - 1
    norm_grid[:, :, :, 1] = norm_grid[:, :, :, 1] / h * 2 - 1

    return grid_sample(data, norm_grid, mode='bilinear')



"""
Support utils
"""


def rad2deg(radians):
    return radians * 180 / np.pi


def vec_to_cross(vec):
    C = torch.zeros((vec.shape[0], 3, 3)).to(vec.device)

    C[:, 0, 1] = -vec[:, 2].squeeze()
    C[:, 0, 2] = vec[:, 1].squeeze()
    C[:, 1, 0] = vec[:, 2].squeeze()
    C[:, 1, 2] = -vec[:, 0].squeeze()
    C[:, 2, 0] = -vec[:, 1].squeeze()
    C[:, 2, 1] = vec[:, 0].squeeze()

    return C


def rotate_a_b_axis_angle(a, b):
    a = a / np.clip(np.linalg.norm(a), a_min=1e-16, a_max=None)
    b = b / np.clip(np.linalg.norm(b), a_min=1e-16, a_max=None)
    rot_axis = np.cross(a, b)
    #   find a proj onto b
    a_proj = b * (a.dot(b))
    a_ort = a - a_proj
    #   find angle between a and b in [0, np.pi)
    theta = np.arctan2(np.linalg.norm(a_ort), np.linalg.norm(a_proj))
    if a.dot(b) < 0:
        theta = np.pi - theta

    aa = rot_axis / np.clip(np.linalg.norm(rot_axis), a_min=1e-16, a_max=None) * theta
    return aa
