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
import numpy as np

import config
import model as modellib
import data as datalib
import utils

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

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
    config = config.CocoInferenceConfig()

    # Create model object.
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    model.load_state_dict(torch.load(args.model))

    print(model)

    img = skimage.io.imread(args.image)
    if img .ndim != 3:
        img = skimage.color.grey2rgb(img)

    # Run detection
    class_ids, scores, boxes, masks = model.detect(img)

    if class_ids is not None:
        class_names = []
        for i in range(len(class_ids)):
            j = class_ids[i]
            class_names.append(datalib.CocoLabel.name(j))
            print(j, datalib.CocoLabel.zh_name(j), boxes[i], scores[i])

        utils.display_instances(img, np.array(boxes), np.array(masks),
                                np.array(class_ids), class_names, np.array(scores))
    else:
        utils.display_instances(img, None, None, None, class_names, None)
