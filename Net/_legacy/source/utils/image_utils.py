def filter_border(image, radius=4):
    """
    :param image: N x C x H x W
    :param radius: int
    """
    _, _, h, w = image.size()
    r2 = radius * 2

    mask = torch.ones((1, 1, h - r2, w - r2)).to(image.device)
    mask = F.pad(input=mask, pad=[radius, radius, radius, radius, 0, 0, 0, 0])

    return image * mask

def select_score_and_keypoints(score, nms_thresh, k_size, top_k):
    """
    :param score: B x 1 x H x W
    :param nms_thresh: float
    :param k_size: int
    :param top_k: int
    :return B x 1 x H x W, B x N x 2
    """
    n, c, h, w = score.size()

    score = filter_border(score)

    # Apply nms
    score = nms(score, nms_thresh, k_size)

    # Extract maximum activation indices and convert them to keypoints
    score = score.view(n, c, -1)
    _, flat_ids = torch.topk(score, top_k)
    keypoints = flat2grid(flat_ids, w).squeeze(1)

    # Select maximum activations
    kp_score_mask = torch.zeros_like(score).to(score.device)
    kp_score_mask = kp_score_mask.scatter(dim=-1, index=flat_ids, value=1)

    score = score * kp_score_mask
    score = score.view(n, c, h, w)

    return score, keypoints




def erode_filter(mask):
    """
    :param mask: N x 1 x H x W
    """
    morph_ellipse_kernel = torch.tensor([[[[0, 0, 1, 0, 0],
                                           [1, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 1],
                                           [1, 1, 1, 1, 1],
                                           [0, 0, 1, 0, 0]]]]).type_as(mask).to(mask.device)

    morphed_mask = apply_kernel(mask, morph_ellipse_kernel) / morph_ellipse_kernel.sum()
    morphed_mask = morphed_mask.ge(0.8).float()

    return morphed_mask


def dilate_filter(mask, ks=3):
    """
    :param mask: N x 1 x H x W
    :param ks: dilate kernel size
    """
    dilate_kernel = torch.ones((1, 1, ks, ks)).type_as(mask).to(mask.device)

    dilated_mask = apply_kernel(mask, dilate_kernel).gt(0).float()

    return dilated_mask


def create_desc_center_coordinates_grid(out, grid_size, permute=True):
    """
    :param out: B x C x H x W
    :param grid_size: int
    :param permute: bool
    """
    coo_grid = create_coordinates_grid(out)
    coo_grid = coo_grid * grid_size + grid_size // 2
    if permute:
        coo_grid = coo_grid[:, :, :, [1, 0]]

    return coo_grid