# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 08月 19日 星期日 20:38:18 CST
# ***
# ************************************************************************************/

import argparse

import os
import random
import skimage.io

import coco
import model as modellib
import utils

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
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

zh_class_names = [
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

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


parser = argparse.ArgumentParser(description='Mask RCNN Predictor')
parser.add_argument(
    '-model',
    type=str,
    default=COCO_MODEL_PATH,
    help='trained model [' + COCO_MODEL_PATH + ']')
parser.add_argument('image', type=str, help='image file')

if __name__ == '__main__':
    random.seed()

    args = parser.parse_args()
    config = InferenceConfig()

    # Create model object.
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    model.load_state_dict(torch.load(args.model))

    print(model)

    img = skimage.io.imread(args.image)

    # Run detection
    results = model.detect([img])

    r = results[0]
    for i in range(len(r['rois'])):
        print(r['class_ids'][i], zh_class_names[r['class_ids'][i]], r['rois'][i], r['scores'][i])

    utils.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
