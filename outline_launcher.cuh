#pragma once

#include <cuda_runtime.h>
#include "effects.cuh"

__host__ void outlines(dim3 gridDim, dim3 blockDim, cudaStream_t stream, unsigned char* d_img, unsigned char* d_img_copy, int width, int height, int shiftX, int shiftY);