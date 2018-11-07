/************************************************************************************
***
*** File Author: Dell, 2018-11-06 17:11:09
***
************************************************************************************/

#include <torch/extension.h>


at::Tensor nms_cuda(const at::Tensor boxes, float threshold);

void crop_gpu_forward(
    const at::Tensor& image,
    const at::Tensor& boxes,           // [y1, x1, y2, x2]
    const at::Tensor& box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    at::Tensor& crops
);

void crop_gpu_backward(
    const at::Tensor& grads,
    const at::Tensor& boxes,      // [y1, x1, y2, x2]
    const at::Tensor& box_index,    // range in [0, batch_size)
    at::Tensor& grads_image // resize to [bsize, c, hc, wc]
);

