#pragma once
#include "effects.cuh"

void radialBlur_CUDA(
	const dim3 gridDim, const dim3 blockDim, const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_imgCopy, const int width, const int height,
	const float centerX, const float centerY, const int blurRadius, const float intensity
);

void radialBlurRGBA_CUDA(
	const dim3 gridDim, const dim3 blockDim, const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_imgCopy, const int width, const int height,
	const float centerX, const float centerY, const int blurRadius, const float intensity
);
