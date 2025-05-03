#include "softPalette_launcher.cuh"

void softPalette(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsBGR, const int numColors
) {
	blendNearestColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, colorsBGR, numColors);
}

void softPaletteRGBA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsRGB, const int numColors
) {
	blendNearestColorsRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, colorsRGB, numColors);
}
