#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void censor_kernel(unsigned char* img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void roundColors_kernel(unsigned char* img, int rows, int cols, int thresh);
__global__ void horizontalLine_kernel(unsigned char* img, int rows, int cols, int lineWidth, int thresh);
__global__ void dynamicColor_kernel(unsigned char* img, int rows, int cols, const unsigned char* colors_BGR, int num_colors);
