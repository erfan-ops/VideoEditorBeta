#pragma once

#include <cuda_runtime.h>
#include "effects.cuh"

__host__ void censor_CUDA(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	int width, int height, int pixelWidth, int pixelHeight
);

__host__ void censorRGBA_CUDA(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	int width, int height, int pixelWidth, int pixelHeight
);
