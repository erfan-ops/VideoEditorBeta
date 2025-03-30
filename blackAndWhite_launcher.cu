#include "hueShift_launcher.cuh"

void blackAndWhite(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int nPixels, const float middle
) {
	blackNwhite_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, middle);
}
