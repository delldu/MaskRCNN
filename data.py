# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 12月 26日 星期三 02:21:38 CST
# ***
# ************************************************************************************/

import os
import random
import colorsys
import torch
import torchvision
import numpy as np
import torchvision.datasets as dataset
import torchvision.transforms as transform
from PIL import Image, ImageDraw, ImageFilter
import sys
import pdb


class Box(object):
    """Box Interpreter, (r,c) format, [[y1, x1] --> [y2, x2])."""

    def __init__(self, y1, x1, y2, x2):
        """Create Box."""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def left(self):
        """Box left."""
        return self.x1

    def right(self):
        """Box right."""
        return self.x2

    def top(self):
        """Box top."""
        return self.y1

    def bottom(self):
        """Box bottom."""
        return self.y2

    def height(self):
        """Box height."""
        return self.y2 - self.y1

    def width(self):
        """Box width."""
        return self.x2 - self.x1

    def center(self):
        """Return center."""
        return ((self.y1 + self.y2) / 2, (self.x1 + self.x2) / 2)

    def move(self, dy1, dx1, dy2, dx2):
        """Move box, delta dx1, dy1, dx2, dy2."""
        self.y1 += dy1
        self.x1 += dx1
        self.y2 += dy2
        self.x2 += dx2

    @classmethod
    def fromlist(cls, l):
        """Create box from list."""
        b = Box(l[0], l[1], l[2], l[3])
        return b

    def tolist(self):
        """Transform box to list."""
        return [self.y1, self.x1, self.y2, self.x2]

    def __str__(self):
        """Dump box for human beings."""
        return '(Box top: %d, left: %d, bottom: %d, right: %d)' % (self.y1, self.x1, self.y2, self.x2)

    def __repr__(self):
        """Dump box for programers."""
        return self.__str__()


def even_number(x):
    """Round to x to even number."""
    return (x // 2) * 2


def encode_image(image, min_dim, max_dim):
    """Encode image. Suppose min_dim and max_dim are even numbers."""
    scale = 1
    h = even_number(image.height)
    w = even_number(image.width)
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
        nh = even_number(round(h * scale))
        nw = even_number(round(w * scale))
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
        nh = even_number(round(1.0 / scale * image.height))
        nw = even_number(round(1.0 / scale * image.width))
        image = transform.Resize((nh, nw))(image)
    return image


def encode_masks(masks, scale, cropbox):
    """Encode masks. Here masks are list of np.uint8 arrays with 1/0."""
    if scale == 1:
        return masks

    padding = (cropbox.left(), cropbox.top(), cropbox.left(), cropbox.top())
    for i in range(len(masks)):
        maskimg = Image.fromarray(masks[i]).convert('L')
        h = even_number(maskimg.height)
        w = even_number(maskimg.width)
        nh = even_number(round(h * scale))
        nw = even_number(round(w * scale))

        maskimg = transform.Resize((nh, nw))(maskimg)
        maskimg = transform.Pad(padding)(maskimg)
        masks[i] = np.array(maskimg)

    return masks


def decode_masks(masks, scale, cropbox):
    """Decode masks."""
    if scale == 1:
        return masks

    for i in range(len(masks)):
        # Crop
        maskimg = Image.fromarray(masks[i]).convert('L')
        maskimg = transform.CenterCrop((cropbox.height(), cropbox.width()))(maskimg)
        # Scale
        nh = even_number(round(maskimg.height * 1.0 / scale))
        nw = even_number(round(maskimg.width * 1.0 / scale))
        maskimg = transform.Resize((nh, nw))(maskimg)
        masks[i] = np.array(maskimg)

    return masks


def encode_boxes(boxes, scale, cropbox):
    """Encode boxes. Here boxes are list of [y1, x1, y2, x2)."""
    if scale == 1:
        return boxes

    for i in range(len(boxes)):
        boxes[i] = list(map(lambda x: int(x * scale), boxes[i]))
        boxes[i][0] += cropbox.top()
        boxes[i][1] += cropbox.left()
        boxes[i][2] += cropbox.top()
        boxes[i][3] += cropbox.left()

    return boxes


def decode_boxes(boxes, scale, cropbox):
    """Decode boxes."""
    if scale == 1:
        return boxes

    for i in range(len(boxes)):
        boxes[i][0] -= cropbox.top()
        boxes[i][1] -= cropbox.left()
        boxes[i][2] -= cropbox.top()
        boxes[i][3] -= cropbox.left()
        boxes[i] = list(map(lambda x: int(x * 1.0 / scale), boxes[i]))

    return boxes


def blend_mask(image, mask, color):
    """Blend mask. image: PIL format, mask: np.array. color: (r, g, b)."""
    colorimg = Image.new("RGB", image.size, color)
    maskimg = Image.fromarray(mask * 255).convert('L')

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


def random_colors(nums, bright=True):
    """Generate colors from HSV space to RGB."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / nums, 1, brightness) for i in range(nums)]
    fcolors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = []
    for (r, g, b) in fcolors:
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    random.shuffle(colors)
    return colors


def blend_image(image, class_names, boxes, masks, scores=None):
    """Blend image with class_names, boxes, masks and optional scores."""
    colors = random_colors(len(class_names))
    fusion = image
    for i in range(len(masks)):
        fusion = blend_mask(fusion, masks[i], colors[i])

    draw = ImageDraw.Draw(fusion)
    for i in range(len(boxes)):
        b = Box.fromlist(boxes[i])
        draw.rectangle((b.left(), b.top(), b.right(), b.bottom()), None, colors[i])
        label = class_names[i]
        if scores:
            label += " {:.3f}".format(scores[i])
        draw.text((b.left(), b.top()), label, colors[i])
    del draw
    return fusion


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
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_memsize(i, seen) for i in obj])
    return size


def coco_class_name(index):
    """Get coco class name. class_names come from dump_class_names in CocoMaskRCNNDataset."""
    class_names = [
        'BG',
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    return class_names[index] if index < len(class_names) else "BG"


class CocoMaskRCNNDataset(dataset.CocoDetection):
    """Coco Dataset for Mask RCNN."""

    def __init__(self, root, annfile):
        """Initialize CocoMaskRCNNDataset."""
        super(CocoMaskRCNNDataset, self).__init__(root, annfile)
        self.min_dim = 800
        self.max_dim = 1024

    def __getitem__(self, index):
        """Get image, labels.
        image: CxHxW, torch.float32, [0, 1.0]
        class_ids: N, torch.int64
        boxes: Nx4, torch.int64
        masks: NxHxW, torch.uint8
        """
        img, class_ids, class_names, boxes, masks = self.load(index)

        image, scale, cropbox = encode_image(img, self.min_dim, self.max_dim)
        masks = encode_masks(masks, scale, cropbox)
        boxes = encode_boxes(boxes, scale, cropbox)

        image = transform.ToTensor()(image)
        class_ids = torch.from_numpy(np.array(class_ids))
        boxes = torch.from_numpy(np.stack(boxes))
        masks = torch.from_numpy(np.stack(masks))

        pdb.set_trace()

        # xxxx3333

        # Convert
        # images = torch.from_numpy(images.transpose(2, 0, 1)).float()
        # image_metas = torch.from_numpy(image_metas)
        # rpn_match = torch.from_numpy(rpn_match)
        # rpn_bbox = torch.from_numpy(rpn_bbox).float()
        # gt_class_ids = torch.from_numpy(gt_class_ids)
        # gt_boxes = torch.from_numpy(gt_boxes).float()
        # gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()

        # # pdb.set_trace()
        # # (Pdb) images.shape
        # # torch.Size([3, 1024, 1024])
        # # (Pdb) gt_masks.shape
        # # torch.Size([4, 56, 56])

        # return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks

        return image, class_ids, boxes, masks

    def image_id(self, index):
        """Image ID."""
        return self.ids[index]

    def image_name(self, index):
        """Image file name."""
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return os.path.join(self.root, path)

    def class_name(self, class_id):
        """Get class name."""
        if class_id in self.coco.cats:
            v = self.coco.cats.get(class_id)
            return v.get("name", "BG")
        return "BG"

    def dump_class_names(self):
        class_names = ['BG']
        for k, v in self.coco.cats.items():
            class_names.append(v["name"])
        print(class_names)

    def load(self, index):
        """Load basic coco image data."""
        image, anns = super().__getitem__(index)

        class_ids = []
        masks = []
        boxes = []
        class_names = []
        for ann in anns:
            class_id = ann['category_id']

            if not class_id:
                print("Warning: NO any object labeled in image {}, {}".format(
                    self.image_id(index), self.image_name(index)))
                continue

            m = self.coco.annToMask(ann)

            # Make sure same size between m and image
            if m.shape[0] != image.height or m.shape[1] != image.width:
                continue

            # Skip small objects
            if m.sum() < 9:
                continue

            if ann['iscrowd']:
                class_id *= -1

            hindex = np.where(np.any(m, axis=0))[0]
            vindex = np.where(np.any(m, axis=1))[0]
            if hindex.shape[0]:
                y1, y2 = vindex[[0, -1]]
                x1, x2 = hindex[[0, -1]]
                # x2 and y2 should not be part of the box.
                x2 += 1
                y2 += 1
            else:
                x1, x2, y1, y2 = 0, 0, 0, 0

            # Skip big objects
            if x2 - x1 >= image.width - 5 or y2 - y1 >= image.height - 5:
                continue

            class_ids.append(class_id)
            boxes.append([y1, x1, y2, x2])
            masks.append(m)
            class_names.append(self.class_name(class_id))

        return image, class_ids, class_names, boxes, masks

    def show(self, index):
        """Show coco raw image."""
        image, class_ids, class_names, boxes, masks = self.load(index)
        blend_image(image, class_names, boxes, masks).show()

    def scale_show(self, index):
        """Get image, labels."""
        img, class_ids, class_names, boxes, masks = self.load(index)
        image, scale, cropbox = encode_image(img, self.min_dim, self.max_dim)
        masks = encode_masks(masks, scale, cropbox)
        boxes = encode_boxes(boxes, scale, cropbox)
        blend_image(image, class_names, boxes, masks).show()

    def memory_show(self):
        """Show memory size."""
        n = get_memsize(self)
        print("Dataset spend memory", n, "bytes, about", n // (1024 * 1024), "M.")


def test():
    root = "data/val2014"
    annfile = "data/annotations/instances_valminusminival2014.json"
    valid_dataset = CocoMaskRCNNDataset(root, annfile)


    # for i in range(len(valid_dataset)):
    #     valid_dataset[i]

    valid_dataset[0]
    # .show(117)
    # pdb.set_trace()


test()
