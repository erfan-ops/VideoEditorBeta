#include "trueOutlines_launcher.cuh"

__host__ void trueOutlines_CUDA(
	const int gridSize, const int blockSize,
	const dim3 gridDim, const dim3 blockDim,
	const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_img_copy,
	const int width, const int height, const int nPixels, const int thresh
) {
	fastBlur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, height, width, thresh);
	subtract_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, d_img_copy, nPixels);
}

__host__ void trueOutlinesRGBA_CUDA(
	const int gridSize, const int blockSize,
	const dim3 gridDim, const dim3 blockDim,
	const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_img_copy,
	const int width, const int height, const int nPixels, const int thresh
) {
	fastBlurRGBA_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, height, width, thresh);
	subtractRGBA_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, d_img_copy, nPixels);
}
