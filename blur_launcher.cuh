#pragma once
#include "effects.cuh"

__host__ void blur_CUDA(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_img_copy,
	int width, int height, int blurRadius
);

__host__ void blurRGBA_CUDA(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_img_copy,
	int width, int height, int blurRadius
);
