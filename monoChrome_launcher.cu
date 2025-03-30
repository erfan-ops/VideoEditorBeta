#include "monoChrome_launcher.cuh"

void monoChrome(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels) {
	monoChrome_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels);
}
