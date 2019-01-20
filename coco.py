"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import time
import numpy as np
import config
import pdb
import data as datalib

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  COCO Evaluation
############################################################
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format ."""
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.class_id(class_id),
                "bbox":
                [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            # pdb.set_trace()
            # {
            #     'image_id': 532481,
            #     'category_id': 1,
            #     'bbox': [259, 179, 63, 55],
            #     'score': 0.99867451,
            #     'segmentation': {
            #         'size': [426, 640],
            #         'counts':
            #         b'gP^33U=2O2N2O0O2N2O2M2O3L4L5L2N1O1N2N2O1O0O2N2N100O100O101N1001O0001O00001OO2O000O2N101O5J5L1N2O1N101N1O2N2N1O3L4L5JcmT4'
            #     }
            # }

            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Run official COCO evaluation.

    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        if i % 10 == 0:
            print("Evaluating ", eval_type, " ", i + 1, " ... ")
        # Load image
        image = dataset.load_image(image_id)

        # print(image_id, dataset.image_name(dataset.image_index(image_id)))

        # Run detection
        t = time.time()
        class_ids, scores, boxes, masks = model.detect(image)
        if class_ids is None:
            continue
        t_prediction += (time.time() - t)

        scores = np.array(scores)
        boxes = np.array(boxes).astype(np.int32)
        masks = np.array(masks).astype(np.uint8)
        masks = masks.transpose(1, 2, 0)     # NxHxW--> HxWxN

        image_results = build_coco_results(dataset, image_ids[i:i + 1],
                                           boxes, class_ids,
                                           scores, masks)
        # pdb.set_trace()

        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train/Eval Mask R-CNN Model on MS COCO.')
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument(
        '--dataset',
        required=True,
        metavar="/path/to/coco/",
        help='Directory of the MS-COCO dataset')
    parser.add_argument(
        '--year',
        required=False,
        default=DEFAULT_DATASET_YEAR,
        metavar="<year>",
        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument(
        '--model',
        required=False,
        default="models/mask_rcnn_coco.pth",
        metavar="/path/to/weights.pth",
        help="Path to weights .pth file or 'coco'")
    parser.add_argument(
        '--logs',
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument(
        '--limit',
        required=False,
        default=500,
        metavar="<image count>",
        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = config.CocoConfig()
    else:
        config = config.CocoInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config, model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights
    print("Loading weights ", args.model)
    model.load_weights(args.model)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.

        dataset_train = datalib.CocoMaskRCNNDataset(args.dataset, "train", args.year, config)
        dataset_valid = datalib.CocoMaskRCNNDataset(args.dataset, "minival", args.year, config)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        model.train_model(
            dataset_train,
            dataset_valid,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads')

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        model.train_model(
            dataset_train,
            dataset_valid,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        model.train_model(
            dataset_train,
            dataset_valid,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=160,
            layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_valid = datalib.CocoMaskRCNNDataset(args.dataset, "minival", args.year, config)

        # dataset_valid.set_filter([532481])

        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_valid, dataset_valid.coco, "bbox", limit=int(args.limit))
        evaluate_coco(model, dataset_valid, dataset_valid.coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
