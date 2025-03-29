#pragma once
#include "effects.cuh"

void radialBlur(
	const dim3 gridDim, const dim3 blockDim, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int width, const int height,
	const float centerX, const float centerY, const int blurRadius, const float intensity
);
