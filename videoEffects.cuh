#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void pixelate_kernel(unsigned char* img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void censor_kernel(unsigned char* img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void roundColors_kernel(unsigned char* img, int rows, int cols, int thresh);
__global__ void horizontalLine_kernel(unsigned char* img, int rows, int cols, int lineWidth, int thresh);
__global__ void dynamicColor_kernel(unsigned char* img, int rows, int cols, const unsigned char* colors_BGR, int num_colors);
__global__ void nearestColor_kernel(unsigned char* img, int rows, int cols, const unsigned char* colors_BGR, int num_colors);
__global__ void radial_blur_kernel(unsigned char* img, int rows, int cols, float centerX, float centerY, int blurRadius, float intensity);
__global__ void reverse_contrast(unsigned char* img, int rows, int cols);
__global__ void shift_hue_kernel(unsigned char* img, int rows, int cols, float rotationFactor);
__global__ void outlines(unsigned char* img, const unsigned char* img_copy, const int rows, const int cols, const int shiftX, const int shiftY);
__global__ void subtract(unsigned char* img1, const unsigned char* img2, const int rows, const int cols);
