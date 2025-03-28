#pragma once

#include <cuda_runtime.h>
#include "effects.cuh"

__host__ void pixelate(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	int width, int height, int pixelWidth, int pixelHeight
);
