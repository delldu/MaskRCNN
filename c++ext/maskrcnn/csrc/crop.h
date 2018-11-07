/************************************************************************************
***
*** File Author: Dell, 2018-11-06 17:11:09
***
************************************************************************************/

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
void crop_forward(
    const at::Tensor& image,
    const at::Tensor& boxes,           // [y1, x1, y2, x2]
    const at::Tensor& box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    at::Tensor& crops
)
{
    if (image.type().is_cuda()) {
#ifdef WITH_CUDA
        crop_gpu_forward(image, boxes, box_index, extrapolation_value, crop_height, crop_width, crops);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else {
        crop_cpu_forward(image, boxes, box_index, extrapolation_value, crop_height, crop_width, crops);
    } 
}

void crop_backward(
    const at::Tensor& grads,
    const at::Tensor& boxes,      // [y1, x1, y2, x2]
    const at::Tensor& box_index,    // range in [0, batch_size)
    at::Tensor& grads_image // resize to [bsize, c, hc, wc]
)
{
    if (grads.type().is_cuda()) {
#ifdef WITH_CUDA
        crop_gpu_backward(grads, boxes, box_index, grads_image);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else {
        crop_cpu_backward(grads, boxes, box_index, grads_image);
    }
}
