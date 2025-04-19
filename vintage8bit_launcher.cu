#include "vintage8bit_launcher.cuh"

__host__ void vintage8bit(
	const dim3 gridDim, const dim3 blockDim, const int gridSize, const int blockSize, const int roundGridSize, cudaStream_t stream,
	unsigned char* __restrict d_img,
	const int pixelWidth, const int pixelHeight, const int thresh, const unsigned char* d_colors_BGR, const int nColors,
	int width, int height, int nPixels, int size
) {
	//dynamicColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, d_colors_BGR, nColors);
	nearestColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, d_colors_BGR, nColors);
	censor_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, pixelWidth, pixelHeight);
	roundColors_kernel<<<roundGridSize, blockSize, 0, stream>>>(d_img, size, thresh);
	// horizontalLine_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, lineWidth, lineDarkeningThresh);
}
