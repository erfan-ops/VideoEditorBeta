#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void monoChrome_kernel(unsigned char* img, int rows, int cols, const unsigned char* color_BGR);
__global__ void censor_kernel(unsigned char* img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void roundColors_kernel(unsigned char* img, int rows, int cols, int thresh);
__global__ void horizontalLine_kernel(unsigned char* img, int rows, int cols, int lineWidth, int thresh);
__global__ void triColor_kernel(
    unsigned char* img,
    int rows,
    int cols,
    const unsigned char* color1_BGR,
    const unsigned char* color2_BGR,
    const unsigned char* color3_BGR
);

__global__ void multi_censor_kernel(unsigned char* batch, int batchSize, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void multi_roundColors_kernel(unsigned char* batch, int batchSize, int rows, int cols, int thresh);
