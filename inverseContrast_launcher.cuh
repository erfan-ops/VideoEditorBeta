#pragma once
#include "effects.cuh"

void inverseContrast_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels);
void inverseContrastRGBA_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels);
