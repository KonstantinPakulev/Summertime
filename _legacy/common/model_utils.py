import kornia

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

vgg_structure = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128]


def make_vgg_block(in_channels, out_channels, kernel_size, padding, activation):
    block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
    block += [nn.BatchNorm2d(out_channels)]

    if activation is not None:
        block += [activation]

    return block


def make_vgg_backbone():
    layers = []
    in_channels = 3
    for v in vgg_structure:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += make_vgg_block(in_channels, v, 3, 1, nn.ReLU(inplace=True))
            in_channels = v

    return nn.Sequential(*layers)


def make_detector_head(config):
    layers = []
    layers += make_vgg_block(vgg_structure[-1], 256, 3, 1, nn.ReLU(inplace=True))
    layers += make_vgg_block(256, 1 + pow(config['grid_size'], 2), 1, 0, None)

    return nn.Sequential(*layers)


def make_descriptor_head(config):
    layers = []
    layers += make_vgg_block(vgg_structure[-1], 256, 3, 1, nn.ReLU(inplace=True))
    layers += make_vgg_block(256, config['descriptor_size'], 1, 0, None)

    return nn.Sequential(*layers)


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, x):
        output = x.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


def space_to_depth(x, block_size):
    block_size_sq = block_size * block_size

    output = x.permute(0, 2, 3, 1)
    (batch_size, s_height, s_width, s_depth) = output.size()

    d_depth = s_depth * block_size_sq
    d_width = int(s_width / block_size)
    d_height = int(s_height / block_size)

    t_1 = output.split(block_size, 2)
    stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]

    output = torch.stack(stack, 1)
    output = output.permute(0, 2, 1, 3)
    output = output.permute(0, 3, 1, 2)

    return output


def detector_loss(logits, keypoint_map, mask, device, config):
    # Convert key point map to feature space and empty bin for no-detection case
    labels = keypoint_map[:, None, :, :]
    labels = space_to_depth(labels, config['grid_size'])
    empty_bin = torch.ones([labels.size(0), 1, labels.size(2), labels.size(3)], dtype=torch.float32).to(device)
    labels = torch.argmax(torch.cat([labels, empty_bin], dim=1), dim=1)

    mask = space_to_depth(mask, config['grid_size'])
    mask = mask.prod(dim=1)

    loss = (cross_entropy(logits, labels, reduction='none') * mask).mean()

    return loss


def descriptor_loss(descriptors, warped_descriptors, homography, mask, device, config):
    batch_size = descriptors.shape[0]
    Hc = descriptors.shape[2]
    Wc = descriptors.shape[3]

    # Compute the position of the center pixel of every cell in the image. Shape is (Hc, Wc, 2)
    # coord_cells is now a grid containing the coordinates of the Hc x Wc
    # center pixels of the 8x8 cells of the image
    coord_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=-1).float()
    coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2

    # Compute the position of the warped center pixels
    warped_coord_cells = warp_points_torch(coord_cells.view(-1, 2), homography)
    # warped_coord_cells is now a list of the warped coordinates of all the center
    # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

    # Compute the pairwise distances and filter the ones less than a threshold
    # The distance is just the pairwise norm of the difference of the two grids
    # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
    coord_cells = coord_cells.reshape([1, 1, 1, Hc, Wc, 2])
    warped_coord_cells = warped_coord_cells.reshape([batch_size, Hc, Wc, 1, 1, 2])

    cell_distances = torch.norm(coord_cells - warped_coord_cells, dim=-1)
    s = torch.where(cell_distances <= config['grid_size'] - 0.5, torch.ones_like(cell_distances, dtype=torch.float), torch.zeros_like(cell_distances, dtype=torch.float)).to(device)
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['grid_size']
    # and 0 otherwise

    # Compute the pairwise dot product between descriptors: d^t * d'
    descriptors = descriptors.view([batch_size, Hc, Wc, 1, 1, -1])
    warped_descriptors = warped_descriptors.view([batch_size, 1, 1, Hc, Wc, -1])
    dp_desc = torch.sum(descriptors * warped_descriptors, dim=-1)

    # dp_desc[id_batch, h, w, h', w'] is the dot product between the
    # descriptor at position (h, w) in the original descriptors map and the
    # descriptor at position (h', w') in the warped image

    # Compute the loss
    positive_dist = (config['positive_margin'] - dp_desc).clamp(min=0.0)
    negative_dist = (dp_desc - config['negative_margin']).clamp(min=0.0)
    loss = config['lambda_d'] * s * positive_dist + (torch.tensor(1, dtype=torch.float).to(device) - s) * negative_dist

    # Mask the pixels if bordering artifacts appear
    mask = space_to_depth(mask, config['grid_size'])
    mask = torch.prod(mask, dim=1)
    mask = mask.reshape([batch_size, 1, 1, Hc, Wc])

    normalization = torch.sum(mask) * Hc * Wc
    loss = torch.sum(mask * loss) / normalization

    return loss


def detector_metrics(probs, y):
    precision = torch.sum(probs * y) / torch.sum(probs)
    recall = torch.sum(probs * y) / torch.sum(y)

    return {'precision': precision, 'recall': recall}


def homography_adaptation(image, probs, model, device, config):
    probs = probs.unsqueeze(0).unsqueeze(0)
    probs[probs.le(config['detection_threshold'])] = 0

    for i in range(config['homography_adaptation']['num']):
        flat_homography = sample_homography(image.shape[2:], **config['homography_adaptation']['homographies'])

        warped_image = kornia.warp_perspective(image, torch.tensor(flat2mat(flat_homography), dtype=torch.float).to(device),
                                               dsize=(image.shape[2], image.shape[3]))
        warped_prob = model(warped_image)['probs'].unsqueeze(0).unsqueeze(0)

        unwarped_prob = kornia.warp_perspective(warped_prob, torch.tensor(flat2mat(invert_homography(flat_homography)), dtype=torch.float).to(device),
                                                dsize=(image.shape[2], image.shape[3]))
        unwarped_prob[unwarped_prob.le(config['detection_threshold'])] = 0

        probs = torch.cat((probs, unwarped_prob))

    if config['homography_adaptation']['aggregation'] == 'sum':
        probs = probs.squeeze().sum(dim=0)

    if 'nms' in config and config['nms']:
        probs = non_maximum_supression(probs, config['nms'], config['iou_threshold'], config['top_k'])

    return probs


def filter_probabilities(probs, config):
    probs[probs.le(config['detection_threshold'])] = 0

    if 'nms' in config and config['nms']:
        probs = non_maximum_supression(probs, config['nms'], config['iou_threshold'], config['top_k'])

    return probs


def non_maximum_supression(probs, size, iou_threshold, top_k):
    coords = probs.nonzero()
    size = size / 2
    boxes = torch.cat([coords - size, coords + size], dim=1)

    nz_probs = probs[coords[:, 0], coords[:, 1]]

    if boxes.numel() == 0:
        return probs

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = torch.mul(x2 - x1, y2 - y1)

    sorted_probs, ids = nz_probs.sort(0)

    ids = ids[-top_k:]
    count = 0
    kept = torch.zeros_like(nz_probs) if top_k == 0 else torch.zeros(top_k)

    xx1 = boxes.clone()
    yy1 = boxes.clone()
    xx2 = boxes.clone()
    yy2 = boxes.clone()
    w = boxes.clone()
    h = boxes.clone()

    while ids.numel() > 0:
        i = ids[-1]

        kept[count] = i
        count += 1

        ids = ids[:-1]

        torch.index_select(x1, 0, ids, out=xx1)
        torch.index_select(y1, 0, ids, out=yy1)
        torch.index_select(x2, 0, ids, out=xx2)
        torch.index_select(y2, 0, ids, out=yy2)

        xx1 = torch.clamp(xx1, min=x1[i].item())
        yy1 = torch.clamp(yy1, min=y1[i].item())
        xx2 = torch.clamp(xx2, max=x2[i].item())
        yy2 = torch.clamp(yy2, max=y2[i].item())

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = xx2 - xx1
        h = yy2 - yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        inter = w * h

        rem_areas = torch.index_select(area, 0, ids)
        union = (rem_areas - inter) + area[i]

        IoU = inter.float() / union.float()

        ids = ids[IoU.le(iou_threshold)]

    kept = kept[:count].long()
    nms_coords = coords[kept]
    nms_probs = nz_probs[kept]

    result = torch.zeros_like(probs)
    result[nms_coords[:, 0], nms_coords[:, 1]] = nms_probs

    return result


# class MegaDepthHomoGroundTruthDataset(Dataset):
#
#     def __init__(self, dataset_root, scene_info_root,
#                  mode, train_ratio,
#                  item_transforms=None, sources=False):
#         self.dataset_root = dataset_root
#         self.sources = sources
#
#         self.dataset = []
#         self.item_transforms = item_transforms
#
#         dataset_path = os.path.join(scene_info_root, "gt_matches.p")
#
#         if os.path.exists(dataset_path):
#             with open(dataset_path, 'rb') as file:
#                 self.dataset = pickle.load(file)
#         else:
#             scene_info_names = os.listdir(scene_info_root)
#             scene_info_names = [name for name in scene_info_names if name.endswith(".npz")]
#
#             for scene_info_name in scene_info_names:
#                 scene_info_path = os.path.join(scene_info_root, scene_info_name)
#
#                 if not os.path.exists(scene_info_path):
#                     print("Scene path doesn't exist:", scene_info_path)
#                     continue
#
#                 scene_info = np.load(scene_info_path, allow_pickle=True)
#
#                 overlap_matrix = scene_info['overlap_matrix']
#                 scale_ratio_matrix = scene_info['scale_ratio_matrix']
#
#                 valid = np.logical_and(
#                     np.logical_and(overlap_matrix >= MIN_OVERLAP_RATIO,
#                                    overlap_matrix <= MAX_OVERLAP_RATIO),
#                     scale_ratio_matrix <= MAX_SCALE_RATIO)
#
#                 pairs = np.vstack(np.where(valid))
#
#                 image_paths = scene_info['image_paths']
#
#                 points3D_id_to_2D = scene_info['points3D_id_to_2D']
#                 points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
#
#                 for pair_idx in range(pairs.shape[1]):
#                     idx1 = pairs[0, pair_idx]
#                     idx2 = pairs[1, pair_idx]
#
#                     matches = np.array(list(
#                         points3D_id_to_2D[idx1].keys() &
#                         points3D_id_to_2D[idx2].keys()
#                     ))
#
#                     # Scale filtering
#                     matches_nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
#                     matches_nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
#                     scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
#                     matches = matches[np.where(scale_ratio <= MAX_SCALE_RATIO)[0]]
#
#                     point2D1 = np.array([points3D_id_to_2D[idx1][m] for m in matches])
#                     point2D2 = np.array([points3D_id_to_2D[idx2][m] for m in matches])
#
#                     homo12, _ = cv2.findHomography(point2D1, point2D2, cv2.RANSAC, 1, confidence=0.999)
#                     homo21, _ = cv2.findHomography(point2D2, point2D1, cv2.RANSAC, 1, confidence=0.999)
#
#                     self.dataset.append([os.path.join(self.dataset_root, image_paths[idx1]),
#                                          os.path.join(self.dataset_root, image_paths[idx2]),
#                                          homo12, homo21])
#
#         with open(dataset_path, 'wb') as file:
#             pickle.dump(self.dataset, file)
#
#         train_split = int(len(self.dataset) * train_ratio)
#
#         if mode == TRAIN:
#             self.dataset = self.dataset[:train_split]
#         elif mode == VAL:
#             self.dataset = self.dataset[train_split:]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         image1_path, image2_path, homo12, homo21 = self.dataset[index]
#
#         image1 = io.imread(image1_path)
#         image2 = io.imread(image2_path)
#
#         item = {IMAGE1_NAME: image1_path.split("/")[-1],
#                 IMAGE2_NAME: image2_path.split("/")[-1],
#                 IMAGE1: image1, IMAGE2: image2,
#                 HOMO12: homo12, HOMO21: homo21}
#
#         if self.sources:
#             item[S_IMAGE1] = image1.copy()
#             item[S_IMAGE2] = image2.copy()
#
#         if self.item_transforms is not None:
#             item = self.item_transforms(item)
#
#         return item
#
#
# class MegaDepthWarpV1GroundTruthDataset(Dataset):
#
#     def __init__(self, dataset_root, item_transforms=None, sources=False):
#         self.dataset_root = dataset_root
#         self.sources = sources
#
#         self.dataset = []
#         self.item_transforms = item_transforms
#
#         dataset_path = os.path.join(dataset_root, "train_list/landscape/imgs_MD.p")
#
#         with open(dataset_path, 'rb') as file:
#             self.dataset = pickle.load(file)
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         image1_path = os.path.join(self.dataset_root, self.dataset[index])
#         image1 = io.imread(image1_path)
#
#         item = {IMAGE1_NAME: image1_path.split("/")[-1],
#                 IMAGE2_NAME: image1_path.split("/")[-1] + "_warp",
#                 IMAGE1: image1}
#
#         if self.item_transforms is not None:
#             item = self.item_transforms(item)
#
#         return item
# Mapping from 3D scene points to their 2D projection coordinates onto images
# points3D_id_to_2D = scene_info['points3D_id_to_2D']
# # Mapiing from 3D scene points to their estimated depth value on images
# points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']

# 3D points projected onto images idx1 and idx2
# proj_idx1 = points3D_id_to_2D[idx1]
# proj_idx2 = points3D_id_to_2D[idx2]
#
# # 3D points which have projections on both images
# matches = np.array(list(proj_idx1.keys() & proj_idx2.keys()))
#
# # Depth of 3D points on images idx1 and idx2
# depth_idx1 = points3D_id_to_ndepth[idx1]
# depth_idx2 = points3D_id_to_ndepth[idx2]
#
# # Scale filtering
# matches_nd1 = np.array([depth_idx1[match] for match in matches])
# matches_nd2 = np.array([depth_idx2[match] for match in matches])
# scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
#
# matches = matches[np.where(scale_ratio <= MAX_SCALE_RATIO)[0]]
#
# points2D_idx1 = np.array([proj_idx1[match] for match in matches])
# points2D_idx2 = np.array([proj_idx2[match] for match in matches])






