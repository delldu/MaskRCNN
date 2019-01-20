# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 12月 17日 星期一 16:33:18 CST
# ***
# ************************************************************************************/

import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

RESIZE_WIDTH = 320
RESIZE_HEIGHT = 480


def resize_grabcut(img):
    """Image Resize and Grab Cut."""
    img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    mask = np.zeros((img.shape[:2]), np.uint8)
    bgmodel = np.zeros((1, 65), np.float64)
    fgmodel = np.zeros((1, 65), np.float64)

    border = random.randint(10, 15)
    rect = (border, border, img.shape[1] - border, img.shape[0] - border)

    cv2.grabCut(img, mask, rect, bgmodel, fgmodel, 16, cv2.GC_INIT_WITH_RECT)

    # 0 -- cv2.GC_BGD, 1 -- cv2.GC_FGD, 2 -- cv2.GC_PR_BGD, 3 -- cv2.GC_PR_FGD
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return img * mask2[:, :, np.newaxis]


def images_show(image1, image2, title1=None, title2=None):
    """Show images."""
    plt.subplot(1, 2, 1)
    plt.title('image 1' if not title1 else title1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('image 2' if not title2 else title2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    plt.show()


def grabcut(image_filename):
    """Two stages image grab cut."""
    image = cv2.imread(image_filename)
    image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))

    result = resize_grabcut(image)

    images_show(image, result, 'Orignal', 'GrabCut')

if __name__ == '__main__':
    grabcut(sys.argv[1])
