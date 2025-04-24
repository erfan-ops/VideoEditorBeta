#include "posterize_launcher.cuh"

__host__ void posterize(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int size, const float threshold
) {
	roundColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, size, threshold);
}

__host__ void posterizeRGBA(
	const int gridSize, const int blockSize, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int size, const float threshold
) {
	roundColors_kernel << <gridSize, blockSize, 0, stream >> > (d_img, size, threshold);
}
