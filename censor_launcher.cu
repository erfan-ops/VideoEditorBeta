#include "censor_launcher.cuh"

__host__ void censor(dim3 gridDim, dim3 blockDim, cudaStream_t stream,
	unsigned char* d_img,
	int width, int height, int pixelWidth, int pixelHeight
) {
	censor_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, height, width, pixelWidth, pixelHeight);
}
