#include "changePalette_launcher.cuh"

void changePalette(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsBGR, const int numColors
) {
	nearestColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, colorsBGR, numColors);
}

void changePaletteRGBA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const unsigned char* __restrict colorsRGB, const int numColors
) {
	nearestColorRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, colorsRGB, numColors);
}
