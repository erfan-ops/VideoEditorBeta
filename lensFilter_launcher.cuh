#pragma once
#include "effects.cuh"

void lensFilter_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int size, const float* __restrict passThreshValues
);

void lensFilterRGBA_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int size, const float* __restrict passThreshValues
);
