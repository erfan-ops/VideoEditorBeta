#include "hueShift_launcher.cuh"

void hueShift_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const float hue, const float saturation, const float lightness
) {
	shift_hue_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, hue, saturation, lightness);
}

void hueShiftRGBA_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels,
	const float hue, const float saturation, const float lightness
) {
	shiftHueRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, hue, saturation, lightness);
}
