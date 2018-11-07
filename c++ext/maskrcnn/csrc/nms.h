/************************************************************************************
***
*** File Author: Dell, 2018-11-06 17:11:09
***
************************************************************************************/


#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor nms(const at::Tensor& dets, const float threshold)
{
    if (dets.type().is_cuda()) {
  #ifdef WITH_CUDA
        // TODO raise error if not compiled with CUDA
        if (dets.numel() == 0)
            return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
        return nms_cuda(dets, threshold);
  #else
        AT_ERROR("Not compiled with GPU support");
  #endif
    }
    // else
    at::Tensor result = nms_cpu(dets, threshold);
    return result;
}
