"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import utils

import pdb
import maskrcnn

############################################################
#  Logging Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


def progress(loc, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        loc         - Required  : current location (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (loc / float(total)))
    filled_length = int(length * loc // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\n')
    # Print New Line on Complete
    if loc == total:
        print()


def intersect1d(tensor1, tensor2):
    # xxxx3333 torch.IntTensor(list(set(tensor1.tolist()) & set(tensor2.tolist())))
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

class SamePad2d(nn.Module):
    """'SAME' padding.
    """
    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        in_width = input.size(2)
        in_height = input.size(3)
        out_width = math.ceil(float(in_width) / float(self.stride))
        out_height = math.ceil(float(in_height) / float(self.stride))
        pad_along_width = ((out_width - 1) * self.stride + self.kernel_size - in_width)
        pad_along_height = ((out_height - 1) * self.stride + self.kernel_size - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


############################################################
#  FPN Graph
############################################################
class FPN(nn.Module):
    """See the paper "Feature Pyramid Networks for Object Detection" for more details."""

    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(
            1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(
            256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        # x -- torch.Size([1, 3, 1024, 1024])

        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x

        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        # p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        # p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        # p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)

        p4_out = self.P4_conv1(c4_out) + F.interpolate(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.interpolate(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.interpolate(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        # pdb.set_trace()
        # (Pdb) p p2_out.size(), p3_out.size(), p4_out.size(), p5_out.size(), p6_out.size()
        # (torch.Size([1, 256, 256, 256]), torch.Size([1, 256, 128, 128]), torch.Size([1, 256, 64, 64]), 
        # torch.Size([1, 256, 32, 32]), torch.Size([1, 256, 16, 16]))

        return [p2_out, p3_out, p4_out, p5_out, p6_out]


############################################################
#  Resnet Graph
############################################################


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, architecture, stage5=False):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(
                self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None
        # pdb.set_trace()


    def forward(self, x):
        pdb.set_trace()

        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        if self.C5:
            x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride),
                nn.BatchNorm2d(
                    planes * block.expansion, eps=0.001, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


############################################################
#  Proposal Layer
############################################################
def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    # pdb.set_trace()
    # (Pdb) p boxes.size(), deltas.size()
    # (torch.Size([6000, 4]), torch.Size([6000, 4]))
    # y2 - y1
    height = boxes[:, 2] - boxes[:, 0]
    # x2 - x1
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
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


def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)

    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors,
                   config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Currently only supports batchsize 1
    # pdb.set_trace()
    # inputs = [rpn_class, rpn_bbox]
    # (Pdb) inputs[0].size()
    # torch.Size([1, 261888, 2])
    # (Pdb) inputs[1].size()
    # torch.Size([1, 261888, 4])
    # proposal_count = 250

    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    # config.RPN_BBOX_STD_DEV[0.1,  0.1,  0.2,  0.2]
    std_dev = torch.Tensor(config.RPN_BBOX_STD_DEV).view(1, 4)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(500, anchors.size(0))
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :]  # TODO: Support batch size > 1 ff.
    anchors = anchors[order.data, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = [0, 0, height, width]
    boxes = clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    keep = maskrcnn.nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.

    norm = torch.FloatTensor([height, width, height, width])
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)
    # pdb.set_trace()
    # (Pdb) normalized_boxes.size()
    # torch.Size([1, 250, 4])

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################
def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - inputs = [boxes] + feature_map !!!
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Currently only supports batchsize 1
    # pdb.set_trace()
    # inputs -- list(), len(inputs) = 5, inputs[0].shape => torch.Size([100, 4])
    # (Pdb) len(inputs)
    # 5
    # (Pdb) inputs[0].size()
    # torch.Size([1, 250, 4]) -- boxes ...
    # (Pdb) inputs[1].size()
    # torch.Size([1, 256, 256, 256])
    # (Pdb) inputs[2].size()
    # torch.Size([1, 256, 128, 128])
    # (Pdb) inputs[3].size()
    # torch.Size([1, 256, 64, 64])
    # (Pdb) inputs[4].size()
    # torch.Size([1, 256, 32, 32])


    # pool_size = 7
    # image_shape = array([1024, 1024,    3])
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4

    image_area = torch.FloatTensor([float(image_shape[0] * image_shape[1])])

    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + torch.log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)

    # (Pdb) roi_level.size()
    # torch.Size([250, 1])

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    # range(2, 6) -- [2,3,4,5]
    for i, level in enumerate(range(2, 6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = torch.zeros(level_boxes.size(0)).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
        # pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(
        #     feature_maps[i], level_boxes, ind)

        pooled_features = maskrcnn.CropFunction(pool_size, pool_size, 0)(
            feature_maps[i], level_boxes, ind)

        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    # pdb.set_trace()
    # (Pdb) pooled.size()
    # torch.Size([250, 256, 7, 7])

    return pooled


############################################################
#  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    # pdb.set_trace()
    # (Pdb) boxes1.size()
    # torch.Size([500, 4])
    # (Pdb) boxes2.size()
    # torch.Size([3, 4])

    boxes1_repeat = boxes2.size(0)
    boxes2_repeat = boxes1.size(0)
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)
    # boxes1 -- (4500, 4)
    # boxes2 -- (4500, 4)

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
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    # pdb.set_trace() ?????

    # (Pdb) overlaps.size()
    # torch.Size([375, 4])

    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsample proposals and generates target box refinment, class_ids, and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    # Currently only supports batchsize 1
    # pdb.set_trace()
    # (Pdb) proposals.size()
    # torch.Size([1, 386, 4])

    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    # BugFixed: if torch.nonzero(gt_class_ids < 0)
    if torch.nonzero(gt_class_ids < 0).size(0) > 0:
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]

        crowd_boxes = gt_boxes[crowd_ix.data, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[non_crowd_ix.data, :]
        gt_masks = gt_masks[non_crowd_ix.data, :]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = torch.ByteTensor(proposals.size(0) * [True])
        if config.GPU_COUNT:
            no_crowd_bool = no_crowd_bool.cuda()

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    # Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).size(0) > 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]
        # config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO == 33
        positive_count = int(
            config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size(0))
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size(0)
        positive_rois = proposals[positive_indices.data, :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement(positive_rois.data, roi_gt_boxes.data)
        std_dev = torch.Tensor(config.BBOX_STD_DEV).view(1, 4)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data, :, :]

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = torch.arange(roi_masks.size(0)).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()

        masks = maskrcnn.CropFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1],
                                  0)(roi_masks.unsqueeze(1), boxes, box_ids).data
        masks = masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    # BugFixed: torch.nonzero(negative_roi_bool) and positive_count > 0:
    if torch.nonzero(negative_roi_bool).size(0) > 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size(0))
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size(0)
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = torch.zeros(negative_count).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = torch.zeros(negative_count, 4)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1])
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = torch.zeros(negative_count)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = torch.zeros(negative_count, 4).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1])
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
    else:
        rois = torch.FloatTensor()
        roi_gt_class_ids = torch.IntTensor()
        deltas = torch.FloatTensor()
        masks = torch.FloatTensor()
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()

    # pdb.set_trace()
    return rois, roi_gt_class_ids, deltas, masks


############################################################
#  Detection Layer
############################################################

def clip_to_window_(boxes, window):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0].clamp_(float(window[0]), float(window[2]))
    boxes[:, 1].clamp_(float(window[1]), float(window[3]))
    boxes[:, 2].clamp_(float(window[0]), float(window[2]))
    boxes[:, 3].clamp_(float(window[1]), float(window[3]))

def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    # pdb.set_trace()
    # (Pdb) p rois.size()
    # torch.Size([1, 250, 4])

    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = torch.from_numpy(np.array([height, width, height, width])).float()
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale

    # Clip boxes to image window
    clip_to_window_(refined_rois, window)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep_bool = class_ids > 0

    # Filter out low confidence boxes
    # DETECTION_MIN_CONFIDENCE = 0.7
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:, 0]

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep.data]
    pre_nms_scores = class_scores[keep.data]
    pre_nms_rois = refined_rois[keep.data]

    for i, class_id in enumerate(torch.unique(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

        # Sort
        ix_rois = pre_nms_rois[ixs.data]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data, :]

        class_keep = maskrcnn.nms(
            torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data,
            config.DETECTION_NMS_THRESHOLD)

        # Map indicies
        class_keep = keep[ixs[order[class_keep].data].data]

        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = torch.unique(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = torch.cat(
        (refined_rois[keep.data], class_ids[keep.data].unsqueeze(1).float(),
         class_scores[keep.data].unsqueeze(1)),
        dim=1)

    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    # Currently only supports batchsize 1
    # pdb.set_trace()
    # (Pdb) p rois.size()
    # torch.Size([1, 250, 4])
    # (Pdb) p mrcnn_class.size()
    # torch.Size([250, 81]))
    # (Pdb) p mrcnn_bbox.size()
    # torch.Size([250, 81, 4]))
    # (Pdb) p type(image_meta), image_meta.shape
    # (<class 'numpy.ndarray'>, (1, 89))

    rois = rois.squeeze(0)

    _, _, window = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window,
                                   config)

    return detections


############################################################
#  Region Proposal Network
############################################################
class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        # anchors_per_location = 3
        self.anchor_stride = anchor_stride # 1
        self.depth = depth # 256

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) type(x), x.size()
        # (<class 'torch.Tensor'>, torch.Size([1, 256, 256, 256]))

        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        # pdb.set_trace()
        # (Pdb) type(rpn_class_logits), rpn_class_logits.size()
        # (<class 'torch.Tensor'>, torch.Size([1, 196608, 2]))
        # (Pdb) rpn_bbox.size()
        # torch.Size([1, 196608, 4])

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Feature Pyramid Network Heads
############################################################
class Classifier(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

        # pdb.set_trace()
        # (Pdb) a
        # self = Classifier(
        #   (conv1): Conv2d(256, 1024, kernel_size=(7, 7), stride=(1, 1))
        #   (bn1): BatchNorm2d(1024, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        #   (conv2): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        #   (bn2): BatchNorm2d(1024, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        #   (relu): ReLU(inplace), xxxx9999 !!!
        #   (linear_class): Linear(in_features=1024, out_features=81, bias=True)
        #   (softmax): Softmax()
        #   (linear_bbox): Linear(in_features=1024, out_features=324, bias=True)
        # )
        # depth = 256
        # pool_size = 7
        # image_shape = array([1024, 1024,    3])
        # num_classes = 81

    def forward(self, x, rois):
        # (Pdb) p type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size()
        # (<class 'list'>, 4, torch.Size([1, 256, 256, 256]), torch.Size([1, 256, 128, 128]),
        # torch.Size([1, 256, 64, 64]), torch.Size([1, 256, 32, 32]))

        # (Pdb) p rois
        # ( 0 ,.,.) =
        #   0.4157  0.3690  0.5200  0.4814
        #   0.4312  0.3839  0.5355  0.5006
        #   0.4411  0.3984  0.5488  0.5063
        #                ⋮
        #   0.3398  0.3097  0.5855  0.5250
        #   0.6279  0.6666  0.6364  0.6771
        #   0.6220  0.9668  0.6400  0.9874
        # [torch.cuda.FloatTensor of size (1,250,4) (GPU 0)]

        # y = [rois] + x
        # (Pdb) p type(y), len(y), y[0].size(), y[1].size(), y[2].size(), y[3].size(), y[4].size()
        # (<class 'list'>, 5, torch.Size([1, 250, 4])
        # torch.Size([1, 256, 256, 256]),
        # torch.Size([1, 256, 128, 128]), torch.Size([1, 256, 64, 64]), torch.Size([1, 256, 32, 32]))

        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        # pdb.set_trace()

        # (Pdb) type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size()
        # (<class 'list'>, 4, torch.Size([1, 256, 256, 256]), torch.Size([1, 256, 128, 128]),
        # torch.Size([1, 256, 64, 64]), torch.Size([1, 256, 32, 32]))

        # (Pdb) rois.size()
        # torch.Size([1, 2, 4])

        # ---- MASK_POOL_SIZE = 14 ------ !!!
        # (Pdb) self.pool_size, self.image_shape
        # (14, array([1024, 1024,    3]))

        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        # pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([2, 81, 28, 28])

        return x


############################################################
#  Loss Functions
############################################################
def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    # pdb.set_trace()
    # (Pdb) rpn_match.size()
    # torch.Size([1, 261888, 1])
    # (Pdb) rpn_class_logits.size()
    # torch.Size([1, 261888, 2])

    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()
    # (Pdb) rpn_match.size()
    # torch.Size([1, 261888])
    # (Pdb) anchor_class
    # tensor([[0, 0, 0,  ..., 0, 0, 0]], device='cuda:0')
    # (Pdb) anchor_class.size()
    # torch.Size([1, 261888])

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:, 0], indices.data[:, 1], :]
    anchor_class = anchor_class[indices.data[:, 0], indices.data[:, 1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    # pdb.set_trace()
    # (Pdb) p target_bbox.size(), rpn_match.size(), rpn_bbox.size()
    # (torch.Size([1, 128, 4]), torch.Size([1, 261888, 1]), torch.Size([1, 261888, 4]))

    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match == 1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:, 0], indices.data[:, 1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0, :rpn_bbox.size(0), :]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    # pdb.set_trace()
    # (Pdb) p target_class_ids.size(), pred_class_logits.size()
    # (torch.Size([100]), torch.Size([100, 81]))

    if target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = torch.FloatTensor([0])
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # pdb.set_trace()
    # (Pdb) p target_bbox.size(), target_class_ids.size(), pred_bbox.size()
    # (torch.Size([100, 4]), torch.Size([100]), torch.Size([100, 81, 4]))

    # BugFixed: target_class_ids.size():
    if target_class_ids.size(0) > 0:
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:, 0].data, :]
        pred_bbox = pred_bbox[indices[:, 0].data, indices[:, 1].data, :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.FloatTensor([0])
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # BugFixed: if target_class_ids.size():
    # pdb.set_trace()
    # (Pdb) p target_masks.size(), target_class_ids.size(), pred_masks.size()
    # (torch.Size([100, 28, 28]), torch.Size([100]), torch.Size([100, 81, 28, 28]))

    if target_class_ids.size(0) > 0:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0].data, :, :]
        y_pred = pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.FloatTensor([0])
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox,
                   target_class_ids, mrcnn_class_logits, target_deltas,
                   mrcnn_bbox, target_mask, mrcnn_mask):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids,
                                                mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids,
                                              mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids,
                                              mrcnn_mask)

    return [
        rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss,
        mrcnn_mask_loss
    ]


############################################################
#  Data Generator
############################################################
def load_image_gt(dataset,
                  config,
                  image_id,
                  augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = utils.resize_mask(mask, scale, padding)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window)

    return image, image_meta, class_ids, bbox, mask


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """

    # pdb.set_trace() ==>
    # image_shape = (1024, 1024, 3)
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
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)
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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config, augment=True):
        """A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.

            dataset: The Dataset object to pick data from
            config: The model config object
            shuffle: If True, shuffles the samples before every epoch
            augment: If True, applies image augmentation to images (currently only
                     horizontal flips are supported)

            Returns a Python generator. Upon calling next() on it, the
            generator returns two lists, inputs and outputs. The containtes
            of the lists differs depending on the received arguments:
            inputs list:
            - images: [batch, H, W, C]
            - image_metas: [batch, size of image meta]
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                        are those of the image unless use_mini_mask is True, in which
                        case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets
                is True then the outputs list contains target class_ids, bbox deltas,
                and masks.
            """
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.config = config
        self.augment = augment

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]

        # p BACKBONE_SHAPES
        # array([[256, 256],
        #        [128, 128],
        #        [ 64,  64],
        #        [ 32,  32],
        #        [ 16,  16]])

        self.anchors = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
            config.BACKBONE_SHAPES, config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        # Get GT bounding boxes and masks for image.
        image_id = self.image_ids[image_index]
        #  (Pdb) image_id
        # 10408

        image, image_metas, gt_class_ids, gt_boxes, gt_masks = \
            load_image_gt(self.dataset, self.config, image_id, augment=self.augment,
                          use_mini_mask=self.config.USE_MINI_MASK)
        # pdb.set_trace()
        # (Pdb) image.shape
        # (1024, 1024, 3)
        # (Pdb) image_metas
        # array([10408,   480,   640,     3,   128,     0,   896,  1024,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1,     1,
        #            1,     1,     1,     1,     1,     1,     1,     1])
        # (Pdb) gt_class_ids
        # array([16, 64, 65, 67], dtype=int32)
        # (Pdb) image_metas.shape
        # (89,)
        # (Pdb) image_metas
        # tensor([10408,   480,   640,     3,   128,     0,   896,  1024,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #             1,     1,     1,     1,     1,     1,     1,     1,     1])
        # (Pdb) gt_boxes
        # array([[ 299,    0,  672,  571],
        #        [ 152,  474,  723, 1024],
        #        [ 752,  197,  895,  478],
        #        [ 693,  531,  882, 1023]], dtype=int32)
        # (Pdb) gt_masks.shape
        # (56, 56, 4)

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            return None

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(
            image.shape, self.anchors, gt_class_ids, gt_boxes, self.config)

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]),
                self.config.MAX_GT_INSTANCES,
                replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        images = mold_image(image.astype(np.float32), self.config)

        # pdb.set_trace()
        # (Pdb) images.shape
        # (1024, 1024, 3)
        # (Pdb) gt_masks.shape
        # (56, 56, 4)

        # Convert
        images = torch.from_numpy(images.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()

        # pdb.set_trace()
        # (Pdb) images.shape
        # torch.Size([3, 1024, 1024])
        # (Pdb) gt_masks.shape
        # torch.Size([4, 56, 56])

        return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks

    def __len__(self):
        return self.image_ids.shape[0]


############################################################
#  MaskRCNN Class
############################################################
class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture. 
        """

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True)
        # print("restnet 101: ", resnet)

        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)
        # print("fpn: ", self.fpn)

        # Generate Anchors
        self.anchors = torch.from_numpy(
            utils.generate_pyramid_anchors(
                config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
                config.BACKBONE_SHAPES, config.BACKBONE_STRIDES,
                config.RPN_ANCHOR_STRIDE)).float()
        # print("anchors: ", self.anchors)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        # RPN
        # RPN_ANCHOR_RATIOS = [0.5, 1, 2], RPN_ANCHOR_STRIDE = 1
        self.rpn = RPN(
            len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)
        # print("RPN: ", self.rpn)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE,
                                     config.NUM_CLASSES)
        # print("FPN Classifier: ", self.classifier)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE,
                         config.NUM_CLASSES)
        # print("FPN Mask: ", self.mask)

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.apply(set_bn_fix)
        # pdb.set_trace()
        # (Pdb) self.anchors.size()
        # torch.Size([261888, 4])


    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform(m.weight)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(
            self.log_dir,
            "mask_rcnn_{}_*epoch*.pth".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")


    def detect(self, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # pdb.set_trace()
        # (Pdb) molded_images[0].shape
        # torch.Size([3, 1024, 1024])

        # (Pdb) p type(images), len(images), type(images[0]), images[0].shape
        # (<class 'list'>, 1, <class 'numpy.ndarray'>, (426, 640, 3))
        # (Pdb) p type(molded_images), molded_images.shape
        # (<class 'numpy.ndarray'>, (1, 1024, 1024, 3))

        # (Pdb) p type(image_metas), image_metas.shape
        # (<class 'numpy.ndarray'>, (1, 89)), 89 = id(1) + shape(2) + window(4) + num_class(81)
        # (Pdb) p type(windows), windows.shape, windows
        # (<class 'numpy.ndarray'>, (1, 4), array([[ 171,    0,  853, 1024]]))


        # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        # pdb.set_trace()

        # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # Run object detection
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')

        # pdb.set_trace()
        # detections , [torch.cuda.FloatTensor of size (1,7,6) (GPU 0)]
        # mrcnn_mask, [torch.cuda.FloatTensor of size (1,7,81,28,28) (GPU 0)]

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })

        # pdb.set_trace()
        # (Pdb) p type(results), len(results), type(results[0]), len(results[0])
        # (<class 'list'>, 1, <class 'dict'>, 4)
        # (Pdb) print(results[0].keys())
        # dict_keys(['rois', 'class_ids', 'scores', 'masks'])

        # pdb.set_trace()

        return results

    def predict(self, input, mode):
        molded_images = input[0]
        image_metas = input[1]

        # pdb.set_trace()
        # (Pdb) input[0].shape
        # torch.Size([1, 3, 1024, 1024])
        # (Pdb) input[1].shape
        # (1, 89)

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)

        # (Pdb) p p2_out.size(), p3_out.size(), p4_out.size(), p5_out.size(), p6_out.size()
        # (torch.Size([1, 256, 256, 256]), torch.Size([1, 256, 128, 128]),
        # torch.Size([1, 256, 64, 64]), torch.Size([1, 256, 32, 32]), torch.Size([1, 256, 16, 16]))

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # input[0] -- (1,3,1024,1024) (GPU 0)], input[1] -- (Pdb) p type(input[1])
        # <class 'numpy.ndarray'> (Pdb) p input[1].shape (1, 89)

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # (Pdb) p rpn_class_logits.size(), rpn_class.size(), rpn_bbox.size()
        # (torch.Size([1, 261888, 2]), torch.Size([1, 261888, 2]), torch.Size([1, 261888, 4]))

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer(
            [rpn_class, rpn_bbox],
            proposal_count=proposal_count,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            anchors=self.anchors,
            config=self.config)

        # pdb.set_trace()
        # (Pdb) p rpn_rois
        # ( 0 ,.,.) =
        #   0.4157  0.3690  0.5200  0.4814
        #   0.4312  0.3839  0.5355  0.5006
        #   0.4411  0.3984  0.5488  0.5063
        #                ⋮
        #   0.3398  0.3097  0.5855  0.5250
        #   0.6279  0.6666  0.6364  0.6771
        #   0.6220  0.9668  0.6400  0.9874
        # [torch.cuda.FloatTensor of size (1,250,4) (GPU 0)]

        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads

            # pdb.set_trace()
            # (Pdb) len(mrcnn_feature_maps)
            # 4
            # (Pdb) mrcnn_feature_maps[0].shape
            # torch.Size([1, 256, 256, 256])
            # (Pdb) mrcnn_feature_maps[1].shape
            # torch.Size([1, 256, 128, 128])
            # (Pdb) mrcnn_feature_maps[2].shape
            # torch.Size([1, 256, 64, 64])
            # (Pdb) mrcnn_feature_maps[3].shape
            # torch.Size([1, 256, 32, 32])

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(
                mrcnn_feature_maps, rpn_rois)


            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class,
                                         mrcnn_bbox, image_metas)

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = torch.from_numpy(np.array([h, w, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :4] / scale

            # Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            # Create masks for detections
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

            # Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)

            return [detections, mrcnn_mask]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]

            # Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = torch.from_numpy(np.array([h, w, h, w])).float()

            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if not rois.size():
                mrcnn_class_logits = torch.FloatTensor()
                mrcnn_class = torch.IntTensor()
                mrcnn_bbox = torch.FloatTensor()
                mrcnn_mask = torch.FloatTensor()
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(
                    mrcnn_feature_maps, rois)

                # Create masks for detections
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)

            return [
                rpn_class_logits, rpn_bbox, target_class_ids,
                mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask,
                mrcnn_mask
            ]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs,
                    layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # From a specific Resnet stage and up
            "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_set = Dataset(train_dataset, self.config, augment=True)

        train_generator = torch.utils.data.DataLoader(
            train_set, batch_size=1, shuffle=True, num_workers=0)
        val_set = Dataset(val_dataset, self.config, augment=True)
        val_generator = torch.utils.data.DataLoader(
            val_set, batch_size=1, shuffle=True, num_workers=0)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch + 1,
                                                     learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [
            param for name, param in self.named_parameters()
            if param.requires_grad and not 'bn' in name
        ]
        trainables_only_bn = [
            param for name, param in self.named_parameters()
            if param.requires_grad and 'bn' in name
        ]
        optimizer = optim.SGD(
            [{
                'params': trainables_wo_bn,
                'weight_decay': self.config.WEIGHT_DECAY
            }, {
                'params': trainables_only_bn
            }],
            lr=learning_rate,
            momentum=self.config.LEARNING_MOMENTUM)

        import os
        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask = self.train_epoch(
                train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask = self.valid_epoch(
                val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([
                loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class,
                loss_mrcnn_bbox, loss_mrcnn_mask
            ])
            self.val_loss_history.append([
                val_loss, val_loss_rpn_class, val_loss_rpn_bbox,
                val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask
            ])
            utils.plot_loss(
                self.loss_history,
                self.val_loss_history,
                save=True,
                log_dir=self.log_dir)

            # Save model
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        step = 0

        optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(
                rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0

            progress(
                step + 1,
                steps,
                prefix="\t{}/{}".format(step + 1, steps),
                suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".
                format(loss.item(),
                       rpn_class_loss.item(),
                       rpn_bbox_loss.item(),
                       mrcnn_class_loss.item(),
                       mrcnn_bbox_loss.item(),
                       mrcnn_mask_loss.item()),
                length=10)

            # Statistics
            loss_sum += loss.item() / steps
            loss_rpn_class_sum += rpn_class_loss.item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum

    def valid_epoch(self, datagenerator, steps):

        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0

        for inputs in datagenerator:
            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            if not target_class_ids.size():
                continue

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(
                rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

            progress(
                step + 1,
                steps,
                prefix="\t{}/{}".format(step + 1, steps),
                suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}"
                .format(loss.item(), rpn_class_loss.item(),
                        rpn_bbox_loss.item(), mrcnn_class_loss.item(),
                        mrcnn_bbox_loss.item(), mrcnn_mask_loss.item()),
                length=10)

            # Statistics
            loss_sum += loss.item() / steps
            loss_rpn_class_sum += rpn_class_loss.item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.item() / steps

            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(0, image.shape, window)
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks


############################################################
#  Data Formatting
############################################################


def compose_image_meta(image_id, image_shape, window):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
   """
    meta = np.array(
        [image_id] +  # size=1
        list(image_shape) +  # size=3
        list(window)  # size=4 (y1, x1, y2, x2) in image cooredinates
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]  # (y1, x1, y2, x2) window of image in in pixels
    return image_id, image_shape, window



def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL
