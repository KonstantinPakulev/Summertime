import torch
from torch.nn import functional as F
from torch.nn import functional as F
from skimage import transform


def warp(score_map, homography):
    n, c, h, w = score_map.size()

    out_h, out_w = h, w
    gy, gx = torch.meshgrid([torch.arange(out_h), torch.arange(out_w)])
    gx, gy = gx.float().unsqueeze(-1), gy.float().unsqueeze(-1)

    ones = gy.new_full(gy.size(), fill_value=1)
    grid = torch.cat((gx, gy, ones), -1)  # (H, W, 3)
    grid = grid.unsqueeze(0)  # (1, H, W, 3)
    grid = grid.repeat(n, 1, 1, 1)  # (B, H, W, 3)
    grid = grid.view(grid.size(0), -1, grid.size(-1))  # (B, H*W, 3)
    grid = grid.permute(0, 2, 1)  # (B, 3, H*W)
    grid = grid.type_as(homography).to(homography.device)

    # (B, 3, 3) matmul (B, 3, H*W) => (B, 3, H*W)
    grid_w = torch.matmul(homography, grid)
    grid_w = grid_w.permute(0, 2, 1)  # (B, H*W, 3)
    grid_w = grid_w.div(grid_w[:, :, 2].unsqueeze(-1) + 1e-8)  # (B, H*W, 3)
    grid_w = grid_w.view(n, out_h, out_w, -1)[:, :, :, :2]  # (B, H, W, 2)
    grid_w[:, :, :, 0] = grid_w[:, :, :, 0].div(w - 1) * 2 - 1
    grid_w[:, :, :, 1] = grid_w[:, :, :, 1].div(h - 1) * 2 - 1

    out_image = F.grid_sample(score_map, grid_w)  # (N, C, H, W)

    return out_image


def filter_border(score_map, radius=8):
    n, c, h, w = score_map.size()

    mask = score_map.new_full((1, 1, h - 2 * radius, w - 2 * radius), fill_value=1)
    pad = [radius, radius, radius, radius, 0, 0, 0, 0]
    mask = F.pad(
        input=mask,
        pad=pad,
        mode="constant",
        value=0,
    )

    score_map = score_map * mask

    return score_map


def nms(score_map, thresh=0.0, k_size=5):
    """
    non maximum depression in each pixel if it is not maximum probability in its k_size*k_size range
    :param score_map: (B, 1, H, W)
    :param thresh: float
    :param k_size: int
    :return: mask (B, 1, H, W)
    """
    dtype, device = score_map.dtype, score_map.device
    batch, channel, height, width = score_map.size()

    score_map = torch.where(score_map < thresh, torch.zeros_like(score_map), score_map)
    pad_size = k_size // 2
    pad = [2 * pad_size, 2 * pad_size, 2 * pad_size, 2 * pad_size, 0, 0]

    score_map_pad = F.pad(
        input=score_map,
        pad=pad,
        mode="constant",
        value=0,
    )

    slice_map = torch.tensor([], dtype=score_map_pad.dtype, device=device)
    for i in range(k_size):
        for j in range(k_size):
            _slice = score_map_pad[:, :, i: height + 2 * pad_size + i, j: width + 2 * pad_size + j]
            slice_map = torch.cat((slice_map, _slice), 1)

    max_slice = slice_map.max(dim=1, keepdim=True)[0]

    center_map = slice_map[:, slice_map.size(1) // 2, :, :].unsqueeze(1)
    mask = torch.ge(center_map, max_slice)

    mask = mask[:, :, pad_size: height + pad_size, pad_size: width + pad_size]

    return mask.type_as(score_map)


def top_k_map(score_map, k=512):
    """
    find the top k maximum pixel probability in a maps
    :param score_map: (B, 1, H, W)
    :param k: int
    :return: mask (B, 1, H, W)
    """
    batch, _, height, width, = score_map.size()
    maps_flat = score_map.view(batch, -1)

    indices = maps_flat.sort(dim=-1, descending=True)[1][:, :k]
    batch_idx = (
        torch.arange(0, batch, dtype=indices.dtype, device=indices.device)
            .unsqueeze(-1)
            .repeat(1, k)
    )

    batch_idx = batch_idx.view(-1).cpu().detach().numpy()
    row_idx = indices.contiguous().view(-1).cpu().detach().numpy()
    batch_indexes = (batch_idx, row_idx)

    topk_mask_flat = torch.zeros(maps_flat.size(), dtype=torch.uint8).to(score_map.device)
    topk_mask_flat[batch_indexes] = 1

    mask = topk_mask_flat.view(batch, -1, height, width)
    return mask


def get_gauss_filter_weight(k_size, sigma):
    """
    generate a gaussian kernel
    :param k_size: int
    :param sigma: float
    :return: numpy(k_size*k_size)
    """
    mu_x = mu_y = k_size // 2
    if sigma == 0:
        psf = torch.zeros((k_size, k_size)).float()
        psf[mu_y, mu_x] = 1.0
    else:
        sigma = torch.tensor(sigma).float()
        x = torch.arange(k_size)[None, :].repeat(k_size, 1).float()
        y = torch.arange(k_size)[:, None].repeat(1, k_size).float()
        psf = torch.exp(
            -((x - mu_x) ** 2 / (2 * sigma ** 2) + (y - mu_y) ** 2 / (2 * sigma ** 2))
        )
    return psf


def im_rescale(im, output_size):
    h, w = im.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size
    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(im, (new_h, new_w), mode="constant")

    return img, h, w, new_w / w, new_h / h
