#include "inverseContrast_launcher.cuh"

void inverseContrast_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels) {
	reverse_contrast<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels);
}

void inverseContrastRGBA_CUDA(const int gridSize, const int blockSize, const cudaStream_t stream, unsigned char* __restrict d_img, const int nPixels) {
	reverseContrastRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, nPixels);
}
