#include "flatLight_launcher.cuh"


__host__ void flatLight_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float lightness
) {
	fixedLightness_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, lightness);
}

__host__ void flatLightRGBA_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float lightness
) {
	fixedLightnessRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, lightness);
}
