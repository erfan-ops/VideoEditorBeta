#pragma once
#include "effects.cuh"

void monoMask_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsBGR, const int numColors
);

void monoMaskRGBA_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsRGB, const int numColors
);
