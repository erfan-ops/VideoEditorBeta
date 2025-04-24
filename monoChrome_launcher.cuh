#pragma once
#include "effects.cuh"

void monoChrome(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels);
void monoChromeRGBA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels);
