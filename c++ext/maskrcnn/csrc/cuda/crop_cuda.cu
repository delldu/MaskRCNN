/************************************************************************************
***
*** File Author: Dell, 2018-11-06 17:11:09
***
************************************************************************************/


#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


__global__ void crop_forward_kernel(
    const int nthreads, const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = boxes_ptr[b * 4];
        const float x1 = boxes_ptr[b * 4 + 1];
        const float y2 = boxes_ptr[b * 4 + 2];
        const float x2 = boxes_ptr[b * 4 + 3];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        const float *pimage = image_ptr + (b_in * depth + d) * image_height * image_width;
        const float top_left = pimage[top_y_index * image_width + left_x_index];
        const float top_right = pimage[top_y_index * image_width + right_x_index];
        const float bottom_left = pimage[bottom_y_index * image_width + left_x_index];
        const float bottom_right = pimage[bottom_y_index * image_width + right_x_index];

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    }
}

__global__ void crop_backward_kernel(
    const int nthreads, const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        // NHWC: out_idx = d + depth * (w + crop_width * (h + crop_height * b))
        // NCHW: out_idx = w + crop_width * (h + crop_height * (d + depth * b))
        int idx = out_idx;
        const int x = idx % crop_width;
        idx /= crop_width;
        const int y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;

        const float y1 = boxes_ptr[b * 4];
        const float x1 = boxes_ptr[b * 4 + 1];
        const float y2 = boxes_ptr[b * 4 + 2];
        const float x2 = boxes_ptr[b * 4 + 3];

        const int b_in = box_ind_ptr[b];
        if (b_in < 0 || b_in >= batch)
        {
            continue;
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                                : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

        const float in_y = (crop_height > 1)
                                ? y1 * (image_height - 1) + y * height_scale
                                : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1)
        {
            continue;
        }

        const float in_x = (crop_width > 1)
                                ? x1 * (image_width - 1) + x * width_scale
                                : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1)
        {
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        float *pimage = grads_image_ptr + (b_in * depth + d) * image_height * image_width;
        const float dtop = (1 - y_lerp) * grads_ptr[out_idx];
        atomicAdd(
            pimage + top_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dtop
        );
        atomicAdd(
            pimage + top_y_index * image_width + right_x_index, 
            x_lerp * dtop
        );

        const float dbottom = y_lerp * grads_ptr[out_idx];
        atomicAdd(
            pimage + bottom_y_index * image_width + left_x_index, 
            (1 - x_lerp) * dbottom
        );
        atomicAdd(
            pimage + bottom_y_index * image_width + right_x_index, 
            x_lerp * dbottom
        );
    }
}


void do_crop_forward(
    const float *image_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float extrapolation_value, float *crops_ptr)
{   
    const int total_count = num_boxes * crop_height * crop_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        crop_forward_kernel<<<block_count, thread_per_block, 0>>>(
            total_count, image_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width,
            crop_height, crop_width, depth, extrapolation_value, crops_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void do_crop_backward(
    const float *grads_ptr, const float *boxes_ptr,
    const int *box_ind_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float *grads_image_ptr)
{   
    const int total_count = num_boxes * crop_height * crop_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        crop_backward_kernel<<<block_count, thread_per_block, 0>>>(
            total_count, grads_ptr, boxes_ptr,
            box_ind_ptr, num_boxes, batch, image_height, image_width,
            crop_height, crop_width, depth, grads_image_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}

void crop_gpu_forward(
    const at::Tensor& image,
    const at::Tensor& boxes,           // [y1, x1, y2, x2]
    const at::Tensor& box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    at::Tensor& crops
)
{
    image.contiguous();
    boxes.contiguous();
    box_index.contiguous();

    const int batch_size = image.size(0);
    const int depth = image.size(1);
    const int image_height = image.size(2);
    const int image_width = image.size(3);

    const int num_boxes = boxes.size(0);

    // init output space
    crops.resize_({num_boxes, depth, crop_height, crop_width});
    crops.zero_(); // contiguous();

    do_crop_forward(
        image.data<float>(),
        boxes.data<float>(),
        box_index.data<int32_t>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value,
        crops.data<float>()
    );
}


void crop_gpu_backward(
    const at::Tensor& grads,
    const at::Tensor& boxes,      // [y1, x1, y2, x2]
    const at::Tensor& box_index,    // range in [0, batch_size)
    at::Tensor& grads_image // resize to [bsize, c, hc, wc]
)
{
    grads.contiguous();
    boxes.contiguous();
    box_index.contiguous();
    grads_image.contiguous();

    // shape
    const int batch_size = grads_image.size(0);
    const int depth = grads_image.size(1);
    const int image_height = grads_image.size(2);
    const int image_width = grads_image.size(3);

    const int num_boxes = grads.size(0);
    const int crop_height = grads.size(2);
    const int crop_width = grads.size(3);

    // init output space
    grads_image.zero_();

    do_crop_backward(
        grads.data<float>(),
        boxes.data<float>(),
        box_index.data<int32_t>(),
        num_boxes, batch_size, image_height, image_width,
        crop_height, crop_width, depth,
        grads_image.data<float>()
    );
}
