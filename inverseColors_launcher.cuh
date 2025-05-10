#pragma once
#include "effects.cuh"

void inverseColors_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int size);
void inverseColorsRGBA_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int size);
