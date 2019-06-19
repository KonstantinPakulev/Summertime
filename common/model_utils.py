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


def detector_loss(logits, keypoint_map, config):
    # Convert key point map to feature space and empty bin for no-detection case
    labels = keypoint_map[:, None, :, :]
    labels = space_to_depth(labels, config['grid_size'])
    empty_bin = torch.ones([labels.size(0), 1, labels.size(2), labels.size(3)], dtype=torch.float32)
    labels = torch.argmax(torch.cat([labels, empty_bin], dim=1), dim=1)

    loss = cross_entropy(logits, labels)

    return loss


def detector_metrics(probs, y):
    precision = torch.sum(probs * y) / torch.sum(probs)
    recall = torch.sum(probs * y) / torch.sum(y)

    return {'precision': precision, 'recall': recall}

# TODO make it later.
# def non_maximum_suppression(boxes, scores, iou_threshold=0.01, keep_top_k=0):
#     keep = scores.new(scores.size(0)).zero_()
#     if boxes.numel() == 0:
#         return keep
#
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#
#     area = torch.mul(x2 - x1, y2 - y1)
#
#     # Sort in ascending order
#     sorted_scores, ids = scores.sort(0)
#
#     if keep_top_k != 0:
#         ids = ids[-keep_top_k:]
#
#     xx1 = boxes.new()
#     yy1 = boxes.new()
#     xx2 = boxes.new()
#     yy2 = boxes.new()
#     w = boxes.new()
#     h = boxes.new()
#
#     count = 0
#     while ids.numel() > 0:
#         i = ids[-1]
#
#         keep[count] = i
#         count += 1
#
#         if ids.size(0) == 1:
#             break
#
#         ids = ids[:-1]
#
#         torch.index_select(x1, 0, ids, out=xx1)
#         torch.index_select(y1, 0, ids, out=yy1)
#         torch.index_select(x2, 0, ids, out=xx2)
#         torch.index_select(y2, 0, ids, out=yy2)
#
#         xx1 = torch.clamp(xx1, min=x1[i])
#         yy1 = torch.clamp(yy1, min=y1[i])
#         xx2 = torch.clamp(xx2, max=x2[i])
#         yy2 = torch.clamp(yy2, max=y2[i])
#
#         w.resize_as_(xx2)
#         h.resize_as_(yy2)
#         w = xx2 - xx1
#         h = yy2 - yy1
#
#         w = torch.clamp(w, min=0.0)
#         h = torch.clamp(h, min=0.0)
#         inter = w * h
#
#         rem_areas = torch.index_select(area, 0, ids)
#         union = (rem_areas - inter) + area[i]
#         iou = inter / union
#
#         ids = ids[iou.le(iou_threshold)]
#
#     return keep


# def box_nms(probs, size, iou_threshold=0.1, min_prob=0.01, keep_top_k=0):
#     # Get coordinates of points in which probabilities are higher than the threshold
#     points = torch.nonzero((probs >= min_prob))
#     scores = probs[points[:, 0], points[:, 1]]
#
#     # Get coordinates of bounding boxes of these points
#     size = size / 2.
#     boxes = torch.cat([points - size, points + size], dim=1)
#
#     # indices = non_maximum_suppression(boxes, scores, iou_threshold, keep_top_k)
#
#     print(1)






