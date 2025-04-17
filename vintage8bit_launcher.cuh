#pragma once
#include "effects.cuh"

__host__ void vintage8bit(
	const dim3 gridDim, const dim3 blockDim, const int gridSize, const int blockSize, const int roundGridSize, cudaStream_t stream,
	unsigned char* __restrict d_img,
	const int pixelWidth, const int pixelHeight, const int thresh, const unsigned char* d_colors_BGR, const int nColors,
	int width, int height, int nPixels, int size
);
