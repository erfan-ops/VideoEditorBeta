#pragma once
#include "effects.cuh"

__host__ void trueOutlines(
	const int gridSize, const int blockSize,
	const dim3 gridDim, const dim3 blockDim,
	const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_img_copy,
	const int width, const int height, const int nPixels, const int thresh
);

__host__ void trueOutlinesRGBA(
	const int gridSize, const int blockSize,
	const dim3 gridDim, const dim3 blockDim,
	const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_img_copy,
	const int width, const int height, const int nPixels, const int thresh
);
