#pragma once
#include "effects.cuh"

void hueShift(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels, const float rotationFactor
);

void hueShiftRGBA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels, const float rotationFactor
);
