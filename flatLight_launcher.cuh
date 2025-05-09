#pragma once

#include <cuda_runtime.h>
#include "effects.cuh"

__host__ void flatLight_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float lightness
);

__host__ void flatLightRGBA_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float lightness
);
