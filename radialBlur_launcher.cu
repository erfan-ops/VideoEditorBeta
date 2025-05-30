#include "radialBlur_launcher.cuh"

void radialBlur_CUDA(
	const dim3 gridDim, const dim3 blockDim, const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_imgCopy, const int width, const int height,
	const float centerX, const float centerY, const int blurRadius, const float intensity
) {
	radial_blur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_imgCopy, height, width, centerX, centerY, blurRadius, intensity);
}

void radialBlurRGBA_CUDA(
	const dim3 gridDim, const dim3 blockDim, const cudaStream_t stream,
	unsigned char* __restrict d_img, const unsigned char* __restrict d_imgCopy, const int width, const int height,
	const float centerX, const float centerY, const int blurRadius, const float intensity
) {
	radialBlurRGBA_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_imgCopy, height, width, centerX, centerY, blurRadius, intensity);
}
