#include "pixelate_launcher.cuh"

#include <iostream>

__host__ void pixelate_CUDA(
	dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	const int width, const int height,
	const int pixelWidth, const int pixelHeight,
	const int xBound, const int yBound
) {
	pixelate_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, pixelWidth, pixelHeight, xBound, yBound);
}

__host__ void pixelateRGBA_CUDA(
	dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	const int width, const int height,
	const int pixelWidth, const int pixelHeight,
	const int xBound, const int yBound
) {
	pixelateRGBA_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, pixelWidth, pixelHeight, xBound, yBound);
}
