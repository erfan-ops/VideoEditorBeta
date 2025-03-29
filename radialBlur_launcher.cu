#include "radialBlur_launcher.cuh"

void radialBlur(
	const dim3 gridDim, const dim3 blockDim, const cudaStream_t stream,
	unsigned char* __restrict d_img, const int width, const int height,
	const float centerX, const float centerY, const int blurRadius, const float intensity
) {
	radial_blur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, centerX, centerY, blurRadius, intensity);
}
