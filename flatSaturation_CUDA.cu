#include "flatSaturation_CUDA.cuh"


__host__ void flatSaturation_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float saturation
) {
	flatSaturation_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, saturation);
}

__host__ void flatSaturationRGBA_CUDA(int gridSize, int blockSize, cudaStream_t stream,
	unsigned char* d_img,
	int nPixels, float saturation
) {
	flatSaturationRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels, saturation);
}
