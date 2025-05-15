#include "monoMask_launcher.cuh"

void monoMask_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsBGR, const int numColors
) {
	dynamicColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, colorsBGR, numColors);
}

void monoMaskRGBA_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsRGB, const int numColors
) {
	dynamicColorRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, colorsRGB, numColors);
}
