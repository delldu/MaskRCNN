/************************************************************************************
***
*** File Author: Dell, 2018-11-06 17:11:09
***
************************************************************************************/

#include "nms.h"
#include "crop.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms", &nms, "non-maximum suppression");
    m.def("crop_forward", &crop_forward, "crop forward");
    m.def("crop_backward", &crop_backward, "crop backward");
}
