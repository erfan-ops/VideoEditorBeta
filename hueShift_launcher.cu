#include "hueShift_launcher.cuh"

void hueShift(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels, const float rotationFactor) {
	shift_hue_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, rotationFactor);
}

void hueShiftRGBA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels, const float rotationFactor) {
	shiftHueRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, rotationFactor);
}
