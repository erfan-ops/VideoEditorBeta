#include "lensFilter_launcher.cuh"

void lensFilter_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int size, const float* __restrict passThreshValues
) {
	passColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, size, passThreshValues);
}

void lensFilterRGBA_CUDA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int size, const float* __restrict passThreshValues
) {
	passColorsRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, size, passThreshValues);
}
