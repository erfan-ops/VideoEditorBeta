#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void pixelate_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void censor_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void roundColors_kernel(unsigned char* __restrict__ img, const int nPixels, const int thresh);
__global__ void horizontalLine_kernel(unsigned char* __restrict__ img, int rows, int cols, int lineWidth, int thresh);
__global__ void dynamicColor_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* colors_BGR, const int num_colors);
__global__ void nearestColor_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* colors_BGR, const int num_colors);
__global__ void radial_blur_kernel(unsigned char* __restrict__ img, int rows, int cols, float centerX, float centerY, int blurRadius, float intensity);
__global__ void reverse_contrast(unsigned char* __restrict__ img, const int nPixels);
__global__ void shift_hue_kernel(unsigned char* __restrict__ img, const int nPixels, const float rotationFactor);
__global__ void outlines_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int shiftX, const int shiftY);
__global__ void subtract_kernel(unsigned char* __restrict__ img1, const unsigned char* __restrict__ img2, const int nPixels);
__global__ void fastBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius);
__global__ void trueBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius);
__global__ void monoChrome_kernel(unsigned char* __restrict__ img, const int nPixels);
__global__ void passColors_kernel(unsigned char* __restrict__ img, const int nPixels, const float* __restrict__ passThreshValues);
