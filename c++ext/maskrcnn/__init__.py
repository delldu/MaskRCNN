# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018-11-06 11:14:02
# ***
# ************************************************************************************/
#

# torch must be import before maskrcnn._C for symbol parsing
import torch
from torch.autograd import Function

import maskrcnn._C as maskrcnn_cpp

# help(maskrcnn.nms)
# help(maskrcnn.CropFunction)

# import pdb

def nms(dets, threshold):
    return maskrcnn_cpp.nms(dets, threshold)


class CropFunction(Function):
    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value


    def forward(self, image, boxes, box_ind):
        # pdb.set_trace()
        # (Pdb) p image.dtype, boxes.dtype, box_ind.dtype
        # (torch.float32, torch.float32, torch.int32)
        crops = torch.zeros_like(image)

        maskrcnn_cpp.crop_forward(image, boxes, box_ind,
            self.extrapolation_value, self.crop_height, self.crop_width, crops)

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)

        return crops


    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        # pdb.set_trace()
        maskrcnn_cpp.crop_backward(grad_outputs, boxes, box_ind, grad_image)

        return grad_image, None, None
