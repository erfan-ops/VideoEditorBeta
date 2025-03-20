#pragma once

#include <cuda_runtime.h>

__global__ void generateBinaryNoise(unsigned char* __restrict__ img, const int nPixels, size_t seed);