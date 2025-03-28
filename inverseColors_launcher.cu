#include "inverseColors_launcher.cuh"

void inverseColors(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int size) {
	inverseColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, size);
}
