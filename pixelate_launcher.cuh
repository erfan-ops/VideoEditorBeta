#pragma once

#include <cuda_runtime.h>
#include "effects.cuh"

__host__ void pixelate_CUDA(
	dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	const int width, const int height,
	const int pixelWidth, const int pixelHeight,
	const int xBound, const int yBound
);

__host__ void pixelateRGBA_CUDA(
	dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	const int width, const int height,
	const int pixelWidth, const int pixelHeight,
	const int xBound, const int yBound
);
