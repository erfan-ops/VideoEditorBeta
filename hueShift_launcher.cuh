#pragma once
#include "effects.cuh"

void hueShift_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const float hue, const float saturation, const float lightness
);

void hueShiftRGBA_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const float hue, const float saturation, const float lightness
);
