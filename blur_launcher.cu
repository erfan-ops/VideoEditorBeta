#include "blur_launcher.cuh"

__host__ void blur(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img, unsigned char* d_img_copy,
	int width, int height, int blurRadius
) {
	fastBlur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, height, width, blurRadius);
}

__host__ void blurRGBA(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img, unsigned char* d_img_copy,
	int width, int height, int blurRadius
) {
	fastBlurRGBA_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, height, width, blurRadius);
}
