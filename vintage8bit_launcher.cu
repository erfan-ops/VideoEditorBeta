#include "vintage8bit_launcher.cuh"

#include <iostream>

__host__ void vintage8bit_CUDA(
	const dim3 gridDim, const dim3 blockDim, const int gridSize, const int blockSize, const int roundGridSize, cudaStream_t stream,
	unsigned char* __restrict d_img,
	const int pixelWidth, const int pixelHeight, const float thresh, const unsigned char* d_colors_BGR, const int nColors,
	int width, int height, int nPixels, int size
) {
	nearestColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, d_colors_BGR, nColors);
	censor_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, pixelWidth, pixelHeight);
	roundColors_kernel<<<roundGridSize, blockSize, 0, stream>>>(d_img, size, thresh);
}

__host__ void vintage8bitRGBA_CUDA(
	const dim3 gridDim, const dim3 blockDim, const int gridSize, const int blockSize, const int roundGridSize, cudaStream_t stream,
	unsigned char* __restrict d_img,
	const int pixelWidth, const int pixelHeight, const float thresh, const unsigned char* d_colors_RGB, const int nColors,
	int width, int height, int nPixels, int size
) {
	nearestColorRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, d_colors_RGB, nColors);
	censorRGBA_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, pixelWidth, pixelHeight);
	roundColorsRGBA_kernel<<<roundGridSize, blockSize, 0, stream>>>(d_img, size, thresh);
}
