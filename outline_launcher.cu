#include "outline_launcher.cuh"

__host__ void outlines(dim3 gridDim, dim3 blockDim, cudaStream_t stream, unsigned char* d_img, unsigned char* d_img_copy, int width, int height, int shiftX, int shiftY) {
	outlines_kernel<<<gridDim, blockDim>>>(d_img, d_img_copy, height, width, shiftX, shiftY);
}
