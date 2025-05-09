#pragma once
#include "effects.cuh"

void blackAndWhite_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels, const float middle
);

void blackAndWhiteRGBA_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels, const float middle
);
