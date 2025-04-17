#include "magicEye_launcher.cuh"

__host__ void magicEye(
	const int gridSize, const int blockSize, cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_noise,
	const int nPixels, const float middle
) {
	blackNwhite_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, middle);
	subtract_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, d_noise, nPixels);
}

__host__ void binaryNoise(
	const int gridSize, const int blockSize, cudaStream_t stream,
	unsigned char* __restrict d_noise,
	const int nPixels, const unsigned long long seed
) {
	generateBinaryNoise<<<gridSize, blockSize, 0, stream>>>(d_noise, nPixels, seed);
}
