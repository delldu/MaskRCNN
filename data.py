# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 12月 26日 星期三 02:21:38 CST
# ***
# ************************************************************************************/

import os
import sys
import math
import random
import colorsys
import torch
import numpy as np
import torchvision.datasets as dataset
import torchvision.transforms as transform
from PIL import Image, ImageDraw, ImageFilter
import skimage.io
import skimage.color
import utils
import pdb


class Box(object):
    """Box Interpreter, (r,c) format, [[y1, x1] --> [y2, x2])."""

    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3

    def __init__(self, y1, x1, y2, x2):
        """Create Box."""
        self.data = [y1, x1, y2, x2]

    def left(self):
        """Box left."""
        return self.data[self.LEFT]

    def right(self):
        """Box right."""
        return self.data[self.RIGHT]

    def top(self):
        """Box top."""
        return self.data[self.TOP]

    def bottom(self):
        """Box bottom."""
        return self.data[self.BOTTOM]

    def height(self):
        """Box height."""
        return self.data[self.BOTTOM] - self.data[self.TOP]

    def width(self):
        """Box width."""
        return self.data[self.RIGHT] - self.data[self.LEFT]

    def xcenter(self):
        """Return x center."""
        return (self.data[self.LEFT] + self.data[self.RIGHT]) / 2

    def ycenter(self):
        """Return y center."""
        return (self.data[self.TOP] + self.data[self.BOTTOM]) / 2

    @classmethod
    def fromlist(cls, l):
        """Create box from list."""
        b = Box(l[0], l[1], l[2], l[3])
        return b

    def tolist(self):
        """Transform box to list."""
        return self.data

    def __repr__(self):
        """Dump box."""
        return '(Box top: %d, left: %d, bottom: %d, right: %d)' % (
            self.data[self.TOP], self.data[self.LEFT], self.data[self.BOTTOM],
            self.data[self.RIGHT])


def boxes_clamp_(boxes, window):
    """Clamp boxes. boxes: [N, (y1, x1, y2, x2)] window: [y1, x1, y2, x2]."""
    box = Box.fromlist(window)
    boxes[:, Box.TOP].clamp_(box.top(), box.bottom())
    boxes[:, Box.LEFT].clamp_(box.left(), box.right())
    boxes[:, Box.BOTTOM].clamp_(box.top(), box.bottom())
    boxes[:, Box.RIGHT].clamp_(box.left(), box.right())


def boxes_scale(boxes, scale):
    """Scale boxes. boxes: [N, (y1, x1, y2, x2)], scale: [sy1, sx1, sy2, sx2]."""
    t = torch.Tensor(scale)
    if boxes.is_cuda:
        t = t.cuda()
    return boxes * t


def boxes_deltas(boxes, gt_boxes):
    """Return deltas = boxes - gt_boxes, boxes and gt_boxes are normal: [N, y1, x1, y2, x2]."""
    height = boxes[:, Box.BOTTOM] - boxes[:, Box.TOP]
    width = boxes[:, Box.RIGHT] - boxes[:, Box.LEFT]
    center_y = boxes[:, Box.TOP] + 0.5 * height
    center_x = boxes[:, Box.LEFT] + 0.5 * width

    gt_height = gt_boxes[:, Box.BOTTOM] - gt_boxes[:, Box.TOP]
    gt_width = gt_boxes[:, Box.RIGHT] - gt_boxes[:, Box.LEFT]
    gt_center_y = gt_boxes[:, Box.TOP] + 0.5 * gt_height
    gt_center_x = gt_boxes[:, Box.LEFT] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


def boxes_refine(boxes, deltas):
    """Return boxes + deltas, boxes: [N, y1, x1, y2, x2], deltas: [N, dy, dx, log(dh), log(dw)]."""
    # y2 - y1
    height = boxes[:, Box.BOTTOM] - boxes[:, Box.TOP]
    # x2 - x1
    width = boxes[:, Box.RIGHT] - boxes[:, Box.LEFT]
    center_y = boxes[:, Box.TOP] + 0.5 * height
    center_x = boxes[:, Box.LEFT] + 0.5 * width

    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width

    # log(dh)
    height *= torch.exp(deltas[:, 2])
    # log(dw)
    width *= torch.exp(deltas[:, 3])

    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def boxes_overlaps(boxes1, boxes2):
    """Compute overlaps between boxes1 and boxes2: [N, (y1, x1, y2, x2)]."""
    boxes_is_numpy_data = False
    if isinstance(boxes1, np.ndarray):
        boxes_is_numpy_data = True
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()

    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    m = boxes1.size(0)
    n = boxes2.size(0)
    boxes1 = boxes1.repeat(1, n).view(-1, 4)
    boxes2 = boxes2.repeat(m, 1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = torch.zeros(y1.size(0))
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(m, n)

    if boxes_is_numpy_data:
        overlaps = overlaps.cpu().numpy()

    return overlaps

def encode_image(image, min_dim, max_dim):
    """Encode image. Suppose min_dim and max_dim are even numbers."""
    scale = 1
    h = image.height
    w = image.width
    window = [0, 0, h, w]

    # Scale up but not down
    scale = max(1, min_dim / min(h, w))

    # Does it exceed max dim?
    image_max = max(h, w)
    if round(image_max * scale) > max_dim:
        scale = max_dim / image_max

    # Resize image ?
    if scale != 1:
        # Get new height and width
        nh = round(h * scale)
        nw = round(w * scale)
        image = transform.Resize((nh, nw))(image)

        # Padding ...
        top_pad = (max_dim - nh) // 2
        bottom_pad = max_dim - nh - top_pad
        left_pad = (max_dim - nw) // 2
        right_pad = max_dim - nw - left_pad
        padding = (left_pad, top_pad, right_pad, bottom_pad)
        image = transform.Pad(padding)(image)
        window = [top_pad, left_pad, nh + top_pad, nw + left_pad]

    cropbox = Box.fromlist(window)
    return image, scale, cropbox


def decode_image(image, scale, cropbox):
    """Decode image."""
    image = transform.CenterCrop((cropbox.height(), cropbox.width()))(image)
    if scale != 1:
        nh = round(1.0 / scale * image.height)
        nw = round(1.0 / scale * image.width)
        image = transform.Resize((nh, nw))(image)
    return image


def normalize_image(image, mean_pixel):
    """Transform image to tensor and sub mean -- Normalized image."""
    image = transform.ToTensor()(image)
    image = image * 255.0
    image[0] = image[0] - mean_pixel[0]
    image[1] = image[1] - mean_pixel[1]
    image[2] = image[2] - mean_pixel[2]
    return image


def encode_masks(masks, scale, cropbox):
    """Encode masks. masks is tensor NxHxW) with 1/0."""
    if scale == 1:
        return masks
    all = []
    padding = (cropbox.left(), cropbox.top(), cropbox.left(), cropbox.top())
    for i in range(masks.size(0)):
        maskimg = Image.fromarray(masks[i].cpu().numpy()).convert('L')
        nh = round(maskimg.height * scale)
        nw = round(maskimg.width * scale)
        maskimg = transform.Resize((nh, nw))(maskimg)
        maskimg = transform.Pad(padding)(maskimg)
        all.append(torch.from_numpy(np.array(maskimg)))
    all = torch.stack(all, dim=0)
    if masks.is_cuda:
        all = all.cuda()
    return all


def decode_masks(masks, scale, cropbox):
    """Decode masks.  masks is tensor NxHxW) with 1/0."""
    if scale == 1:
        return masks
    all = []
    for i in range(masks.size(0)):
        # Crop
        maskimg = Image.fromarray(masks[i].cpu().numpy()).convert('L')
        maskimg = transform.CenterCrop((cropbox.height(),
                                        cropbox.width()))(maskimg)
        # Scale
        nh = round(maskimg.height * 1.0 / scale)
        nw = round(maskimg.width * 1.0 / scale)
        maskimg = transform.Resize((nh, nw))(maskimg)
        all.append(torch.from_numpy(np.array(maskimg)))

    all = torch.stack(all, dim=0)
    if masks.is_cuda:
        all = all.cuda()
    return all


def full_masks(class_id, boxes, masks, height, width):
    """Transform masks[2, 81, 28, 28] to [2, heigh, width]. class_is is N, boxes is Nx4, masks NxHxW."""
    all = []
    for i in range(class_id.size(0)):
        mask = masks[i][class_id[i].item()]*255.0

        box = Box.fromlist(boxes[i].tolist())
        maskimg = Image.fromarray(mask.cpu().numpy()).convert('L')
        maskimg = transform.Resize((int(box.height()), int(box.width())))(maskimg)

        # Padding ...
        top_pad = int(box.top())
        bottom_pad = height - maskimg.height - top_pad
        left_pad = int(box.left())
        right_pad = width - maskimg.width - left_pad
        padding = (left_pad, top_pad, right_pad, bottom_pad)

        maskimg = transform.Pad(padding)(maskimg)

        mask = torch.from_numpy(np.array(maskimg))
        mask = mask > 127

        all.append(mask)
    all = torch.stack(all, dim=0)
    if masks.is_cuda:
        all = all.cuda()

    return all


def encode_boxes(boxes, scale, cropbox):
    """Encode boxes. boxes is tensor Nx4 with element format [y1, x1, y2, x2)."""
    if scale == 1:
        return boxes

    boxes = boxes_scale(boxes, [scale, scale, scale, scale])
    boxes[:, 0] += cropbox.top()
    boxes[:, 1] += cropbox.left()
    boxes[:, 2] += cropbox.top()
    boxes[:, 3] += cropbox.left()

    return boxes


def decode_boxes(boxes, scale, cropbox):
    """Decode boxes."""
    if scale == 1:
        return boxes

    boxes[:, 0] -= cropbox.top()
    boxes[:, 1] -= cropbox.left()
    boxes[:, 2] -= cropbox.top()
    boxes[:, 3] -= cropbox.left()

    scale = 1.0 / (scale + 1e-5)
    boxes = boxes_scale(boxes, [scale, scale, scale, scale])
    return boxes


def random_colors(nums, bright=True, shuffle=True):
    """Generate colors from HSV space to RGB."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / nums, 1, brightness) for i in range(nums)]
    fcolors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = []
    for (r, g, b) in fcolors:
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    if shuffle:
        random.shuffle(colors)
    return colors


def blend_mask(image, mask, color):
    """Blend mask. image: PIL format, mask is tensor HxW. color: (r, g, b)."""
    colorimg = Image.new("RGB", image.size, color)
    maskimg = Image.fromarray(mask.numpy() * 255).convert('L')

    # border is also mask image
    border = maskimg.filter(ImageFilter.CONTOUR).point(lambda i: 255 - i)
    # Image filter bug ?
    # remove border points on left == 0, right == 0, top == 0 and bottom == 0
    pixels = border.load()
    for i in range(border.height):
        pixels[0, i] = 0
        pixels[border.width - 1, i] = 0
    for j in range(border.width):
        pixels[j, 0] = 0
        pixels[j, border.height - 1] = 0

    alphaimg = Image.blend(image, colorimg, 0.2)
    context = Image.composite(alphaimg, image, maskimg)

    # Add border
    return Image.composite(colorimg, context, border)


def blend_image(image, label_names, boxes, masks, scores=None):
    """Blend image with label_names, boxes Nx4, masks NxHxW."""
    m = boxes.size(0)
    if m < 1:
        return image
    colors = random_colors(m)
    fusion = image
    if masks is not None:
        for i in range(masks.size(0)):
            fusion = blend_mask(fusion, masks[i], colors[i])

    draw = ImageDraw.Draw(fusion)
    for i in range(m):
        b = Box.fromlist(boxes[i].tolist())
        draw.rectangle((b.left(), b.top(), b.right(), b.bottom()), None, colors[i])
        label = label_names[i] if isinstance(label_names, list) else ""
        if scores:
            label += " {:.3f}".format(scores[i])
        draw.text((b.left(), b.top()), label, colors[i])
    del draw
    return fusion


def draw_anchors(image, anchors):
    """Draw anchors."""
    colors = random_colors(4)
    # image = transform.Resize((1024, 1024))(image)
    draw = ImageDraw.Draw(image)
    boxes = anchors.tolist()
    n = math.sqrt(len(boxes) / 3)
    c = int((n + 1) * n / 2 * 3)

    for i in range(len(boxes)):
        b = Box.fromlist(boxes[i])
        draw.rectangle((b.xcenter() - 1, b.ycenter() - 1, b.xcenter() + 1,
                        b.ycenter() + 1), None, colors[0])

    for i in range(3):
        b = Box.fromlist(boxes[c + i])
        draw.rectangle((b.left(), b.top(), b.right() + 1, b.bottom()), None,
                       colors[i + 1])

    del draw
    image.show()


def get_memsize(obj, seen=None):
    """Get memory size."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_memsize(v, seen) for v in obj.values()])
        size += sum([get_memsize(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_memsize(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        size += sum([get_memsize(i, seen) for i in obj])
    return size


def rpn_samples(anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [m] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """

    # pdb.set_trace() ==>
    # anchors = array([[ -22.627417  ,  -11.3137085 ,   22.627417  ,   11.3137085 ],
    #        [ -16.        ,  -16.        ,   16.        ,   16.        ],
    #        [ -11.3137085 ,  -22.627417  ,   11.3137085 ,   22.627417  ],
    #        ...,
    #        [ 597.96132803,  778.98066402, 1322.03867197, 1141.01933598],
    #        [ 704.        ,  704.        , 1216.        , 1216.        ],
    #        [ 778.98066402,  597.96132803, 1141.01933598, 1322.03867197]])
    # anchors.shape = (261888, 4)

    # gt_class_ids = array([42, 42, 42, 43, 44, 45, 45, 46, 61], dtype=int32)
    # gt_boxes = array([[ 277,  403,  450,  542],
    #        [ 350,  711,  515,  856],
    #        [ 291,   37,  475,  219],
    #        [ 555,  651,  598,  687],
    #        [ 501,  847,  642, 1023],
    #        [ 392,  144,  485,  259],
    #        [ 377,  354,  448,  448],
    #        [ 565,  832,  659,  949],
    #        [ 224,    0,  792, 1024]], dtype=int32)

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # (Pdb) rpn_match.shape
    # (261888,)

    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    # config.RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = boxes_overlaps(anchors, crowd_boxes)

        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    overlaps = boxes_overlaps(anchors, gt_boxes)

    # (Pdb) p type(anchors), anchors.shape
    # (<class 'numpy.ndarray'>, (261888, 4))
    # (Pdb) p type(gt_boxes), gt_boxes.shape
    # (<class 'numpy.ndarray'>, (17, 4))
    # p overlaps.shape   (261888, 17)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1

    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1

    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (
        config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


class CocoLabel(object):
    """Coco Label. Label must be continues for deep learning, but coco class id is not."""

    @classmethod
    def name(cls, label_id):
        """Label no range is [0, 80]."""
        label_names = [
            'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        return label_names[label_id] if label_id < len(label_names) else "BG"

    @classmethod
    def zh_name(cls, label_id):
        """Label range is [0, 80]."""
        zh_names = [
            '背景', '人', '自行车', '汽车', '摩托车', '飞机', '公共汽车', '火车',
            '卡车', '船', '红绿灯', '消防栓', '停车标志',
            '停车计时器', '长凳', '鸟', '猫', '狗', '马', '羊', '牛',
            '大象', '熊', '斑马', '长颈鹿', '背包', '伞', '手提包',
            '领带', '手提箱', '飞盘', '滑雪', '滑雪板', '运动球', '风筝',
            '棒球棒', '棒球手套', '滑板', '冲浪板',
            '网球拍', '瓶子', '酒杯', '杯子', '叉子', '刀', '勺子',
            '碗', '香蕉', '苹果', '三明治', '橙子', '花椰菜', '胡萝卜',
            '热狗', '比萨饼', '甜甜圈', '蛋糕', '椅子', '沙发', '盆栽植物',
            '床', '餐桌', '厕所', '电视', '笔记本电脑', '鼠标', '遥控器',
            '键盘', '手机', '微波炉', '烤箱', '烤面包机', '水槽',
            '冰箱', '书', '钟', '花瓶', '剪刀', '泰迪熊',
            '吹风机', '牙刷'
        ]

        return zh_names[label_id] if label_id < len(zh_names) else '背景'

    @classmethod
    def from_class(cls, class_id):
        """Class id range is [0, 90]. Remove coco class holes."""
        class_label_map = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
            10: 10, 11: 11, 13: 12, 14: 13, 15: 14,
            16: 15, 17: 16, 18: 17, 19: 18, 20: 19,
            21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
            27: 25, 28: 26, 31: 27, 32: 28, 33: 29,
            34: 30, 35: 31, 36: 32, 37: 33, 38: 34,
            39: 35, 40: 36, 41: 37, 42: 38, 43: 39,
            44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
            50: 45, 51: 46, 52: 47, 53: 48, 54: 49,
            55: 50, 56: 51, 57: 52, 58: 53, 59: 54,
            60: 55, 61: 56, 62: 57, 63: 58, 64: 59,
            65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
            74: 65, 75: 66, 76: 67, 77: 68, 78: 69,
            79: 70, 80: 71, 81: 72, 82: 73, 84: 74,
            85: 75, 86: 76, 87: 77, 88: 78, 89: 79,
            90: 80
        }

        return class_label_map[class_id]

    @classmethod
    def to_class(cls, label_id):
        """Transform label id to class id."""
        class_list = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
            88, 89, 90, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
            85, 86, 87, 88, 89, 90
        ]
        return class_list[label_id] if label_id < len(class_list) else 0


def coco_annfile(dir, subset, year=2014):
    """Construct coco annotation file."""
    annfile = '{}/annotations/instances_{}{}.json'.format(dir, subset, year)
    return annfile


def coco_root(dir, subset, year=2014):
    """Construct coco root dir."""
    if subset == "minival" or subset == "valminusminival":
        subset = "val"
    image_dir = "{}/{}{}".format(dir, subset, year)
    return image_dir


class CocoMaskRCNNDataset(dataset.CocoDetection):
    """Coco Dataset for Mask RCNN."""

    def __init__(self, rootdir, subset, year, config):
        """Initialize CocoMaskRCNNDataset."""
        root = coco_root(rootdir, subset, year)
        annfile = coco_annfile(rootdir, subset, year)
        super(CocoMaskRCNNDataset, self).__init__(root, annfile)

        self.config = config
        self.anchors = utils.create_pyramid_anchors(
            config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
            config.BACKBONE_SHAPES, config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, index):
        """Get image, labels.

        image: CxHxW, torch.float, [0, 1.0]
        label_ids: N, torch.int
        boxes: Nx4, torch.float, y1, x1, y2, x2
        masks: NxHxW, torch.float
        """
        rawimg, label_ids, label_names, boxes, masks = self.load(index, hflip=True)

        image, scale, cropbox = encode_image(rawimg, self.config.IMAGE_MIN_DIM,
                                             self.config.IMAGE_MAX_DIM)
        masks = encode_masks(masks, scale, cropbox)
        boxes = encode_boxes(boxes, scale, cropbox)

        image = normalize_image(image, self.config.MEAN_PIXEL)

        rpn_match, rpn_bbox = rpn_samples(self.anchors, np.array(label_ids),
                                          boxes.numpy(), self.config)
        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()

        # pdb.set_trace()
        # masks = masks.permute(1, 2, 0)     # NxHxW--> HxWxN

        return image, rpn_match, rpn_bbox, label_ids, boxes, masks

    def set_filter(self, image_ids):
        """Set filter for dataset with image ids(list format)."""
        self.ids = image_ids

    def image_id(self, index):
        """Image ID."""
        return self.ids[index]

    def image_index(self, image_id):
        """Search index for image id."""
        return self.ids.index(image_id)

    def image_name(self, index):
        """Image file name."""
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return os.path.join(self.root, path)

    def class_id(self, label_id):
        """Get class id for given label."""
        return CocoLabel.to_class(label_id)

    def load_image(self, image_id):
        """Return image with skimgae.io format."""
        index = self.image_index(image_id)
        path = self.image_name(index)
        img = skimage.io.imread(path)
        if img .ndim != 3:
            img = skimage.color.grey2rgb(img)
        return img

    def show(self, index):
        """Show coco raw image."""
        image, label_ids, label_names, boxes, masks = self.load(index)

        print("Image file ID: ", self.image_id(index), ", name: ", self.image_name(index))
        print("Image size: Height x Width = ", image.height, "x", image.width)
        blend_image(image, label_names, boxes, masks).show()

    def net_show(self, index):
        """Get image, labels."""
        img, label_ids, label_names, boxes, masks = self.load(index, hflip=True)
        image, scale, cropbox = encode_image(img, self.config.IMAGE_MIN_DIM,
                                             self.config.IMAGE_MAX_DIM)
        masks = encode_masks(masks, scale, cropbox)
        boxes = encode_boxes(boxes, scale, cropbox)

        print("Image file ID: ", self.image_id(index), ", name: ", self.image_name(index))
        print("Image size: Height x Width = ", image.height, "x", image.width)

        blend_image(image, label_names, boxes, masks).show()

    def summary(self):
        """Dataset summary."""
        print("Dataset pictures:", super().__len__())
        n = get_memsize(self)
        print("Dataset memory", n, "bytes, about", n // (1024 * 1024), "M.")

    def load(self, index, hflip=False):
        """Load basic coco image data."""
        image, anns = super().__getitem__(index)

        # For enhance data
        hflip = hflip & (random.randint(0, 1))

        if hflip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        label_ids = []
        masks = []
        boxes = []
        label_names = []

        for ann in anns:
            class_id = ann['category_id']

            # Transform class_id to label_id
            label_id = CocoLabel.from_class(class_id)

            if not label_id:
                print("Warning: Invalid label in image {}, {}".format(
                    self.image_id(index), self.image_name(index)))
                continue

            m = self.coco.annToMask(ann)

            # Skip small objects
            if m.max() < 1:
                continue

            if ann['iscrowd']:
                class_id *= -1
                # Make sure same size between m and image
                if m.shape[0] != image.height or m.shape[1] != image.width:
                    m = np.ones([image.height, image.width], dtype=bool)

            if hflip:
                m = np.fliplr(m)

            hindex = np.where(np.any(m, axis=0))[0]
            vindex = np.where(np.any(m, axis=1))[0]
            if hindex.shape[0]:
                y1, y2 = vindex[[0, -1]]
                x1, x2 = hindex[[0, -1]]
                # x2 and y2 should not be part of the box.
                x2 += 1.0
                y2 += 1.0
            else:
                x1, x2, y1, y2 = 0.0, 0.0, 0.0, 0.0

            label_ids.append(label_id)
            boxes.append([y1, x1, y2, x2])
            m = torch.from_numpy(m.astype(np.float))
            masks.append(m)
            label_names.append(CocoLabel.name(label_id))

        if label_ids:
            label_ids = torch.IntTensor(label_ids)
            boxes = torch.Tensor(boxes)
            masks = torch.stack(masks, dim=0)
        else:
            print("Warning ...", index, self.image_name(index))
            # Fake label_ids = [0], Backgroud
            # boxes = [[0, 0, image.height, image.width]]
            # masks = [[1, ..., 1]]
            label_ids = torch.IntTensor([0])
            # image is 'PIL.Image.Image'>
            boxes = torch.Tensor([[0, 0, image.height, image.width]])
            masks = torch.ones(1, image.height, image.width)

        # If more instances than fits in the array, sub-sample from them.
        m = boxes.size(0)
        if m > self.config.MAX_GT_INSTANCES:
            label_ids = label_ids[:m]
            boxes = boxes[:m]
            masks = masks[:m]

        return image, label_ids, label_names, boxes, masks

    def dump_class_names(self):
        """Dump class names for programer reference."""
        class_names = ['BG']
        for k, v in self.coco.cats.items():
            class_names.append(v["name"])
        print(class_names)


def test():
    """Test Coco dataset."""
    import config as configlib
    config = configlib.Config()
    valid_dataset = CocoMaskRCNNDataset("data", "train", 2014, config)

    # valid_dataset.set_filter([44404])
    valid_dataset.net_show(0)
    valid_dataset.summary()

# test()
