"""
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import os
import re
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import utils
import data as datalib

import pdb
import maskrcnn


def progress(loc,
             total,
             prefix='',
             suffix='',
             decimals=1,
             length=100,
             fill='█'):
    """Create terminal progress bar.

    @params:
        loc         - Required  : current location (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (loc / float(total)))
    filled_length = int(length * loc // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\n')
    # Print New Line on Complete
    if loc == total:
        print()


def intersect1d(tensor1, tensor2):
    """Intersect 1D set."""
    x = torch.LongTensor(list(set(tensor1.tolist()) & set(tensor2.tolist())))
    if tensor1.is_cuda:
        x = x.cuda()
    return x


class SamePad2d(nn.Module):
    """'SAME' padding."""

    def __init__(self, kernel_size, stride):
        """Init."""
        super(SamePad2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        """Keep feature size not changing after CNN."""
        in_width = input.size(2)
        in_height = input.size(3)
        out_width = math.ceil(float(in_width) / float(self.stride))
        out_height = math.ceil(float(in_height) / float(self.stride))
        pad_along_width = (
            (out_width - 1) * self.stride + self.kernel_size - in_width)
        pad_along_height = (
            (out_height - 1) * self.stride + self.kernel_size - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        """Dump class name."""
        return self.__class__.__name__


############################################################
#  FPN Graph
############################################################
class FPN(nn.Module):
    """See the paper "Feature Pyramid Networks for Object Detection" for more details."""

    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        """FPN Init."""
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
        self.P3_conv1 = nn.Conv2d(
            512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
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

    def forward(self, x):
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
#  ROIAlign Layer
############################################################
def roi_align(inputs, pool_size, image_shape):
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
    roi_level = 4 + torch.log2(
        torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
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
        feature_maps[i] = feature_maps[i].unsqueeze(0)
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


def mrn_samples(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsample rpn_rois and generates target box deltas, class_ids, and masks for each.

    Inputs:
    rpn_rois: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough rpn_rois.
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
    # (Pdb) rpn_rois.size()
    # torch.Size([1, 386, 4])

    rpn_rois = rpn_rois.squeeze(0)
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

        crowd_overlaps = datalib.boxes_overlaps(rpn_rois, crowd_boxes)

        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = torch.ByteTensor(rpn_rois.size(0) * [True])
        if config.GPU_COUNT:
            no_crowd_bool = no_crowd_bool.cuda()

    overlaps = datalib.boxes_overlaps(rpn_rois, gt_boxes)

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
        positive_rois = rpn_rois[positive_indices.data, :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROIs
        # Why using positive_rois.data, roi_gt_boxes.data instead of positive_rois, roi_gt_boxes ?
        deltas = datalib.boxes_deltas(positive_rois.data, roi_gt_boxes.data)

        std_dev = torch.Tensor(config.BBOX_STD_DEV).view(1, 4)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data, :, :]

        # Compute mask targets
        boxes = positive_rois

        box_ids = torch.arange(roi_masks.size(0)).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()

        masks = maskrcnn.CropFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1],
                                      0)(roi_masks.unsqueeze(1).float(), boxes, box_ids).data
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
        negative_rois = rpn_rois[negative_indices.data, :]
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
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0],
                            config.MASK_SHAPE[1])
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
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0],
                            config.MASK_SHAPE[1])
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

    return rois, roi_gt_class_ids, deltas, masks


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
        rpn_class: [batch, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        # anchors_per_location = 3
        self.anchor_stride = anchor_stride  # 1
        self.depth = depth  # 256

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

        # pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([1, 256, 256, 256])
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))
        # pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([1, 512, 256, 256])

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_class = self.softmax(rpn_class_logits)

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

        return [rpn_class_logits, rpn_class, rpn_bbox]


    def class_loss(self, rpn_match, rpn_class_logits):
        """RPN anchor classifier loss.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive, -1=negative, 0=neutral.
        rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
        """
        # pdb.set_trace()
        # (Pdb) rpn_match.size()
        # torch.Size([1, 261888, 1])
        # (Pdb) rpn_class_logits.size()
        # torch.Size([1, 261888, 2])

        # Squeeze last dim to simplify
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

    def boxes_loss(self, target_bbox, rpn_match, rpn_bbox):
        """Return the RPN bounding box loss.

        target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        """

        # pdb.set_trace()
        # (Pdb) p target_bbox.size(), rpn_match.size(), rpn_bbox.size()
        # (torch.Size([1, 128, 4]), torch.Size([1, 261888, 1]), torch.Size([1, 261888, 4]))

        # Squeeze last dim to simplify
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


############################################################
#  Feature Pyramid Network Heads, Regression would be a better name.
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

        # y = [rois] + x
        # (Pdb) p type(y), len(y), y[0].size(), y[1].size(), y[2].size(), y[3].size(), y[4].size()
        # (<class 'list'>, 5, torch.Size([250, 4])
        # torch.Size([1, 256, 256, 256]),
        # torch.Size([1, 256, 128, 128]), torch.Size([1, 256, 64, 64]), torch.Size([1, 256, 32, 32]))

        # (Pdb) len(x)
        # 4
        # (Pdb) x[0].size()
        # torch.Size([1, 256, 256, 256])
        # (Pdb) x[3].size()
        # torch.Size([1, 256, 32, 32])
        # (Pdb) rois.size()
        # torch.Size([79, 4])
        x = roi_align([rois] + x, self.pool_size, self.image_shape)
        # (Pdb) x.size()
        # torch.Size([79, 256, 7, 7])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        mrn_class_logits = self.linear_class(x)
        mrn_probs = self.softmax(mrn_class_logits)

        mrn_bbox = self.linear_bbox(x)
        mrn_bbox = mrn_bbox.view(mrn_bbox.size(0), -1, 4)

        # pdb.set_trace()
        # (Pdb) mrn_bbox.size()
        # torch.Size([82, 81, 4])

        return [mrn_class_logits, mrn_probs, mrn_bbox]

    def class_loss(self, target_class_ids, pred_class_logits):
        """Loss for the classifier head of Mask RCNN."""
        # pdb.set_trace()
        # (Pdb) p target_class_ids.size(), pred_class_logits.size()
        # (torch.Size([100]), torch.Size([100, 81]))
        if target_class_ids.size(0) > 0:
            loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
        else:
            loss = torch.FloatTensor([0])
            if target_class_ids.is_cuda:
                loss = loss.cuda()

        return loss

    def boxes_loss(self, target_class_ids, target_bbox, pred_bbox):
        """Loss for Mask R-CNN bounding box refinement.

        target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        target_class_ids: [batch, num_rois]. Integer class IDs.
        pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """
        # pdb.set_trace()
        # (Pdb) p target_bbox.size(), target_class_ids.size(), pred_bbox.size()
        # (torch.Size([100, 4]), torch.Size([100]), torch.Size([100, 81, 4]))

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
        # pdb.set_trace()
        # (Pdb) a
        # depth = 256
        # pool_size = 14
        # image_shape = array([1024, 1024, 3])
        # num_classes = 81

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

        x = roi_align([rois] + x, self.pool_size, self.image_shape)
        # pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([79, 256, 14, 14])

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

        # pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([79, 256, 28, 28])

        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        # pdb.set_trace()
        # (Pdb) x.size()
        # torch.Size([79, 81, 28, 28])

        return x

    def mask_loss(self, target_class_ids, target_masks, pred_masks):
        """Mask binary cross-entropy loss for the masks head.

        target_masks: [batch, num_rois, height, width].
            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values from 0 to 1.
        """
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


############################################################
#  Mask RCNN Network -- mrn
############################################################
class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality."""

    def __init__(self, config, model_dir):
        """Global configuration."""
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []
        self.epoch = 0

    def build(self, config):
        """Build Mask R-CNN architecture."""

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. ")

        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()

        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # Generate Anchors
        self.anchors = torch.from_numpy(
            utils.create_pyramid_anchors(
                config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
                config.BACKBONE_SHAPES, config.BACKBONE_STRIDES,
                config.RPN_ANCHOR_STRIDE)).float()

        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        # RPN_ANCHOR_RATIOS = [0.5, 1, 2], RPN_ANCHOR_STRIDE = 1
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE,
                                     config.NUM_CLASSES)

        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE,
                         config.NUM_CLASSES)

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
        """Set model layers as trainable."""
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
        os.makedirs(self.log_dir, exist_ok=True)

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

    def detect(self, image):
        """Run detection pipeline."""
        molded_image, window, scale, padding = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            max_dim=self.config.IMAGE_MAX_DIM,
            padding=self.config.IMAGE_PADDING)
        molded_image = mold_image(molded_image, self.config)

        # pdb.set_trace()
        # (Pdb) molded_image.shape
        # (1024, 1024, 3)

        # Convert images to torch tensor
        molded_image = torch.from_numpy(molded_image.transpose(2, 0, 1)).float()
        molded_image.unsqueeze_(0)
        # molded_image --> (1, 3, 1024, 1024)

        # To GPU
        if self.config.GPU_COUNT:
            molded_image = molded_image.cuda()

        # Run object detection, return tensors with Bx... formats.
        mrn_class_ids, mrn_scores, mrn_boxes, mrn_masks = self.predict(molded_image, window)

        if mrn_class_ids is None:
            return None, None, None, None

        mrn_class_ids.squeeze_(0)
        mrn_scores.squeeze_(0)
        mrn_boxes.squeeze_(0)
        mrn_masks.squeeze_(0)

        # Restore orignal image size ...
        mrn_boxes = datalib.decode_boxes(mrn_boxes, scale, datalib.Box.fromlist(window))
        mrn_masks = datalib.decode_masks(mrn_masks, scale, datalib.Box.fromlist(window))

        # Convert detection results to normal data format.
        mrn_class_ids = mrn_class_ids.cpu().tolist()
        mrn_scores = mrn_scores.detach().cpu().tolist()
        mrn_boxes = mrn_boxes.detach().cpu().tolist()
        mrn_masks = mrn_masks.detach().cpu().tolist()

        return mrn_class_ids, mrn_scores, mrn_boxes, mrn_masks

    def predict(self, molded_image, window):
        """Predict molded_image and return mrn_class_ids, mrn_scores, mrn_boxes, mrn_masks."""
        self.eval()

        # BxCxHxW ..., here B == 1
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_image)

        # (Pdb) p p2_out.size(), p3_out.size(), p4_out.size(), p5_out.size(), p6_out.size()
        # (torch.Size([1, 256, 256, 256]), torch.Size([1, 256, 128, 128]),
        # torch.Size([1, 256, 64, 64]), torch.Size([1, 256, 32, 32]), torch.Size([1, 256, 16, 16]))

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        rpn_class_logits, rpn_class, rpn_bbox = self.rpn_detect(rpn_feature_maps)
        # (Pdb) p rpn_class_logits.size(), rpn_class.size(), rpn_bbox.size()
        # (torch.Size([1, 261888, 2]), torch.Size([1, 261888, 2]), torch.Size([1, 261888, 4]))

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        rpn_rois = self.rpn_refine(rpn_class, rpn_bbox)

        # (Pdb) p rpn_rois
        # [torch.cuda.FloatTensor of size (1,250,4) (GPU 0)]

        # pdb.set_trace()
        # (Pdb) len(mrn_feature_maps)
        # 4
        # (Pdb) mrn_feature_maps[0].shape
        # torch.Size([1, 256, 256, 256])
        # (Pdb) mrn_feature_maps[1].shape
        # torch.Size([1, 256, 128, 128])
        # (Pdb) mrn_feature_maps[2].shape
        # torch.Size([1, 256, 64, 64])
        # (Pdb) mrn_feature_maps[3].shape
        # torch.Size([1, 256, 32, 32])

        # Detections
        mrn_class_logits, mrn_class, mrn_bbox = self.mrn_detect(mrn_feature_maps, rpn_rois)
        mrn_class_ids, mrn_scores, mrn_boxes = self.mrn_refine(rpn_rois, mrn_class, mrn_bbox, window)

        if mrn_class_ids is None:
            return [None, None, None, None]

        # Create masks for detections
        h, w = self.config.IMAGE_SHAPE[:2]
        mrn_rois = mrn_boxes.float() * 1.0 / h
        mrn_masks = self.mask(mrn_feature_maps, mrn_rois)
        mrn_masks = datalib.full_masks(
            mrn_class_ids.squeeze(0),
            mrn_boxes.squeeze(0).detach(), mrn_masks.detach(), h, w).unsqueeze_(0)

        # (Pdb) mrn_class_ids.size()
        # torch.Size([1, 1])
        # (Pdb) mrn_scores.size()
        # torch.Size([1, 1])
        # (Pdb) mrn_boxes.size()
        # torch.Size([1, 1, 4])
        # (Pdb) mrn_masks.size()
        # torch.Size([1, 1, 81, 28, 28])

        return [mrn_class_ids, mrn_scores, mrn_boxes, mrn_masks]

    def extract(self, input):
        """Feature extract."""
        molded_images = input[0]

        # pdb.set_trace()
        # (Pdb) input[0].shape
        # torch.Size([1, 3, 1024, 1024])
        # (Pdb) input[1].shape
        # (1, 89)

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
        mrn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        rpn_class_logits, rpn_class, rpn_bbox = self.rpn_detect(rpn_feature_maps)

        # (Pdb) p rpn_class_logits.size(), rpn_class.size(), rpn_bbox.size()
        # (torch.Size([1, 261888, 2]), torch.Size([1, 261888, 2]), torch.Size([1, 261888, 4]))

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

        gt_class_ids = input[1]
        gt_boxes = input[2]
        gt_masks = input[3]

        # Normalize coordinates
        h, w = self.config.IMAGE_SHAPE[:2]
        gt_boxes = datalib.boxes_scale(gt_boxes, [1.0 / h, 1.0 / w, 1.0 / h, 1.0 / w])

        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        rpn_rois = self.rpn_refine(rpn_class, rpn_bbox)
        rois, target_class_ids, target_deltas, target_mask = mrn_samples(
            rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

        if rois.size(0) < 1:
            mrn_class_logits = torch.FloatTensor()
            mrn_class = torch.IntTensor()
            mrn_bbox = torch.FloatTensor()
            mrn_mask = torch.FloatTensor()
            if self.config.GPU_COUNT:
                mrn_class_logits = mrn_class_logits.cuda()
                mrn_class = mrn_class.cuda()
                mrn_bbox = mrn_bbox.cuda()
                mrn_mask = mrn_mask.cuda()
        else:
            mrn_class_logits, mrn_class, mrn_bbox = self.mrn_detect(mrn_feature_maps, rois)
            mrn_mask = self.mask(mrn_feature_maps, rois)

        extract_result = [
            rpn_class_logits, rpn_bbox, target_class_ids, target_deltas, target_mask,
            mrn_class_logits, mrn_bbox, mrn_mask
        ]

        return extract_result

    def rpn_detect(self, rpn_feature_maps):
        """Rough detect via FPN."""

        # rpn_feature_maps = [p2, p3, p4, p5, p6]
        rpn_layer_outputs = []
        for p in rpn_feature_maps:
            rpn_layer_outputs.append(self.rpn(p))
        rpn_outputs = list(zip(*rpn_layer_outputs))

        rpn_class_logits, rpn_class, rpn_bbox = [torch.cat(list(o), dim=1) for o in rpn_outputs]
        return rpn_class_logits, rpn_class, rpn_bbox


    def rpn_refine(self, rpn_class, rpn_bbox):
        """Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinment detals to anchors.

        Inputs:
           rpn_class: [batch, anchors, (bg prob, fg prob)]
           rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
        """

        # Currently only supports batchsize 1
        # pdb.set_trace()
        # inputs = [rpn_class, rpn_bbox]
        # (Pdb) rpn_class.size()
        # torch.Size([1, 261888, 2])
        # (Pdb) rpn_bbox.size()
        # torch.Size([1, 261888, 4])

        proposal_count = self.config.RPN_NMS_MAX_ROIS_NUM
        nms_threshold = self.config.RPN_NMS_THRESHOLD

        rpn_class = rpn_class.squeeze(0)
        rpn_bbox = rpn_bbox.squeeze(0)

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = rpn_class[:, 1]

        # Box deltas [batch, num_rois, 4]
        # config.RPN_BBOX_STD_DEV[0.1,  0.1,  0.2,  0.2]

        deltas = datalib.boxes_scale(rpn_bbox, self.config.RPN_BBOX_STD_DEV)

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(500, self.anchors.size(0))
        scores, order = scores.sort(descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = deltas[order.data, :]  # TODO: Support batch size > 1 ff.
        anchors = self.anchors[order.data, :]

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = datalib.boxes_refine(anchors, deltas)

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        height, width = self.config.IMAGE_SHAPE[:2]
        datalib.boxes_clamp_(boxes, [0, 0, height, width])
        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        keep = maskrcnn.nms(
            torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
        keep = keep[:proposal_count]
        boxes = boxes[keep, :]

        # Normalize dimensions to range of 0 to 1.

        norm = torch.FloatTensor([height, width, height, width])
        if self.config.GPU_COUNT:
            norm = norm.cuda()
        normalized_boxes = boxes / norm

        # Add back batch dimension
        normalized_boxes = normalized_boxes.unsqueeze(0)
        # pdb.set_trace()
        # (Pdb) normalized_boxes.size()
        # torch.Size([1, 250, 4])

        return normalized_boxes

    def mrn_detect(self, mrn_feature_maps, rois):
        """Mask R-CNN detect."""
        # Return mrn_class_logits, mrn_class, mrn_bbox
        return self.classifier(mrn_feature_maps, rois)

    def mrn_refine(self, rpn_rois, probs, deltas, window):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rpn_rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in image coordinates.

        Return: mrn_class_ids, mrn_scores, mrn_boxes
        """
        # pdb.set_trace()
        # (Pdb) p rpn_rois.size()
        # torch.Size([1, 250, 4])
        rpn_rois = rpn_rois.squeeze(0)

        _, class_ids = torch.max(probs, dim=1)

        # Class probability of the top class of each ROI
        # Class-specific bounding box deltas
        idx = torch.arange(class_ids.size()[0]).long()
        if self.config.GPU_COUNT:
            idx = idx.cuda()
        class_scores = probs[idx, class_ids.data]
        deltas_specific = deltas[idx, class_ids.data]

        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        std_dev = torch.from_numpy(
            np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 4])).float()
        if self.config.GPU_COUNT:
            std_dev = std_dev.cuda()
        refined_rois = datalib.boxes_refine(rpn_rois, deltas_specific * std_dev)

        # Convert coordiates to image domain
        height, width = self.config.IMAGE_SHAPE[:2]
        boxes = datalib.boxes_scale(refined_rois, [height, width, height, width])

        # Clip boxes to image window
        datalib.boxes_clamp_(boxes, window)

        # Round and cast to int since we're deadling with pixels now
        boxes = torch.round(boxes)

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep_bool = class_ids > 0

        # Filter out low confidence boxes
        # DETECTION_MIN_CONFIDENCE = 0.7
        if self.config.DETECTION_MIN_CONFIDENCE:
            keep_bool = keep_bool & (class_scores >= self.config.DETECTION_MIN_CONFIDENCE)
        keep = torch.nonzero(keep_bool)[:, 0]

        if (keep.size(0) < 1):
            # Big Bang !!!
            return None, None, None

        # Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = boxes[keep.data]

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
                self.config.DETECTION_NMS_THRESHOLD)

            # Map indicies
            class_keep = keep[ixs[order[class_keep].data].data]

            if i == 0:
                nms_keep = class_keep
            else:
                nms_keep = torch.unique(torch.cat((nms_keep, class_keep)))
        keep = intersect1d(keep, nms_keep)

        # Keep top detections
        roi_count = self.config.DETECTION_MAX_INSTANCES
        top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
        keep = keep[top_ids.data]

        # Convert boxes to normalized coordinates
        mrn_class_ids = class_ids[keep.data].unsqueeze(0)
        mrn_scores = class_scores[keep.data].unsqueeze(0)
        mrn_boxes = boxes[keep.data].unsqueeze(0)

        return mrn_class_ids, mrn_scores, mrn_boxes


    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers):
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
            "heads":
            r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # From a specific Resnet stage and up
            "3+":
            r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "4+":
            r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "5+":
            r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # All layers
            "all":
            ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0)

        val_generator = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=True, num_workers=0)

        # Train
        print("Starting at epoch {}. LR={}\n".format(self.epoch + 1,
                                                     learning_rate))
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
        optimizer = optim.SGD([{
            'params': trainables_wo_bn,
            'weight_decay': self.config.WEIGHT_DECAY
        }, {
            'params': trainables_only_bn
        }],
                              lr=learning_rate,
                              momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            print("Epoch {}/{}.".format(epoch, epochs))

            # Training
            self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

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
        loss_mrn_class_sum = 0
        loss_mrn_bbox_sum = 0
        loss_mrn_mask_sum = 0
        step = 0

        optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images = inputs[0]
            rpn_match = inputs[1]
            rpn_bbox = inputs[2]
            gt_class_ids = inputs[3]
            gt_boxes = inputs[4]
            gt_masks = inputs[5]

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            results = self.extract([images, gt_class_ids, gt_boxes, gt_masks])

            rpn_class_logits = results[0]
            rpn_pred_bbox = results[1]
            target_class_ids = results[2]
            target_deltas = results[3]
            target_mask = results[4]
            mrn_class_logits = results[5]
            mrn_bbox = results[6]
            mrn_mask = results[7]

            # Compute losses
            rpn_class_loss = self.rpn.class_loss(rpn_match, rpn_class_logits)
            rpn_bbox_loss = self.rpn.boxes_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
            mrn_class_loss = self.classifier.class_loss(target_class_ids, mrn_class_logits)
            mrn_bbox_loss = self.classifier.boxes_loss(target_class_ids, target_deltas, mrn_bbox)
            mrn_mask_loss = self.mask.mask_loss(target_class_ids, target_mask, mrn_mask)

            loss = rpn_class_loss + rpn_bbox_loss + mrn_class_loss + mrn_bbox_loss + mrn_mask_loss

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
                suffix=
                "Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrn_class_loss: {:.5f} - mrn_bbox_loss: {:.5f} - mrn_mask_loss: {:.5f}"
                .format(loss.item(), rpn_class_loss.item(),
                        rpn_bbox_loss.item(), mrn_class_loss.item(),
                        mrn_bbox_loss.item(), mrn_mask_loss.item()),
                length=10)

            # Statistics
            loss_sum += loss.item() / steps
            loss_rpn_class_sum += rpn_class_loss.item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.item() / steps
            loss_mrn_class_sum += mrn_class_loss.item() / steps
            loss_mrn_bbox_sum += mrn_bbox_loss.item() / steps
            loss_mrn_mask_sum += mrn_mask_loss.item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        self.loss_history.append([
            loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrn_class_sum,
            loss_mrn_bbox_sum, loss_mrn_mask_sum
        ])

        return loss_sum

    def valid_epoch(self, datagenerator, steps):
        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrn_class_sum = 0
        loss_mrn_bbox_sum = 0
        loss_mrn_mask_sum = 0

        for inputs in datagenerator:
            images = inputs[0]
            rpn_match = inputs[1]
            rpn_bbox = inputs[2]
            gt_class_ids = inputs[3]
            gt_boxes = inputs[4]
            gt_masks = inputs[5]

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()

            # Run object detection
            results = self.extract([images, gt_class_ids, gt_boxes, gt_masks])
            rpn_class_logits = results[0]
            rpn_pred_bbox = results[1]
            target_class_ids = results[2]
            target_deltas = results[3]
            target_mask = results[4]
            mrn_class_logits = results[5]
            mrn_bbox = results[6]
            mrn_mask = results[7]

            if not target_class_ids.size():
                continue

            # Compute losses
            rpn_class_loss = self.rpn.class_loss(rpn_match, rpn_class_logits)
            rpn_bbox_loss = self.rpn.boxes_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
            mrn_class_loss = self.classifier.class_loss(target_class_ids, mrn_class_logits)
            mrn_bbox_loss = self.classifier.boxes_loss(target_class_ids, target_deltas, mrn_bbox)
            mrn_mask_loss = self.mask.mask_loss(target_class_ids, target_mask, mrn_mask)

            loss = rpn_class_loss + rpn_bbox_loss + mrn_class_loss + mrn_bbox_loss + mrn_mask_loss

            progress(
                step + 1,
                steps,
                prefix="\t{}/{}".format(step + 1, steps),
                suffix=
                "Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrn_class_loss: {:.5f} - mrn_bbox_loss: {:.5f} - mrn_mask_loss: {:.5f}"
                .format(loss.item(), rpn_class_loss.item(),
                        rpn_bbox_loss.item(), mrn_class_loss.item(),
                        mrn_bbox_loss.item(), mrn_mask_loss.item()),
                length=10)

            # Statistics
            loss_sum += loss.item() / steps
            loss_rpn_class_sum += rpn_class_loss.item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.item() / steps
            loss_mrn_class_sum += mrn_class_loss.item() / steps
            loss_mrn_bbox_sum += mrn_bbox_loss.item() / steps
            loss_mrn_mask_sum += mrn_mask_loss.item() / steps

            if step == steps - 1:
                break
            step += 1

        self.val_loss_history.append([
            loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrn_class_sum,
            loss_mrn_bbox_sum, loss_mrn_mask_sum
        ])

        return loss_sum


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order."""
    return images.astype(np.float32) - config.MEAN_PIXEL
