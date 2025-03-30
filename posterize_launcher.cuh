#pragma once
#include "effects.cuh"

__host__ void posterize(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int size, const int threshold
);
