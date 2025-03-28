#include "pixelate_launcher.cuh"

__host__ void pixelate(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	int width, int height, int pixelWidth, int pixelHeight
) {
	pixelate_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, pixelWidth, pixelHeight);
}
