#pragma once

#include <cuda_runtime.h>
#include "effects.cuh"

__host__ void flatSaturation_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float saturation
);

__host__ void flatSaturationRGBA_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float saturation
);
