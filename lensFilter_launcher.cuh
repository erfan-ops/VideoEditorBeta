#pragma once
#include "effects.cuh"

void lensFilter(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int size, const float* __restrict passThreshValues);
