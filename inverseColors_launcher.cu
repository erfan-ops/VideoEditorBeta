#include "inverseColors_launcher.cuh"

void inverseColors_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int size) {
	inverseColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, size);
}

void inverseColorsRGBA_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int size) {
	inverseColorsRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, size);
}
