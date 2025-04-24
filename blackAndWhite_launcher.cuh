#pragma once
#include "effects.cuh"

void blackAndWhite(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels, const float middle
);

void blackAndWhiteRGBA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels, const float middle
);
