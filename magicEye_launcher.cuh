#pragma once
#include "effects.cuh"

__host__ void magicEye(
	const int gridSize, const int blockSize, cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_noise,
	const int nPixels, const float middle
);

__host__ void binaryNoise(
	const int gridSize, const int blockSize, cudaStream_t stream,
	unsigned char* __restrict d_noise,
	const int nPixels, const unsigned long long seed
);
