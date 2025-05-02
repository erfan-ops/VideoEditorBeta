#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void pixelate_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void pixelateRGBA_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void censor_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void censorRGBA_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight);
__global__ void roundColors_kernel(unsigned char* __restrict__ img, const int size, const float thresh);
__global__ void roundColorsRGBA_kernel(unsigned char* __restrict__ img, const int size, const float thresh);
__global__ void horizontalLine_kernel(unsigned char* __restrict__ img, int rows, int cols, int lineWidth, int thresh);
__global__ void dynamicColor_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ colors_BGR, const int num_colors);
__global__ void dynamicColorRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ colors_RGB, const int num_colors);
__global__ void nearestColor_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ colors_BGR, const int num_colors);
__global__ void nearestColorRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ colors_RGB, const int num_colors);
__global__ void radial_blur_kernel(unsigned char* __restrict__ img, int rows, int cols, float centerX, float centerY, int blurRadius, float intensity);
__global__ void radialBlurRGBA_kernel(unsigned char* __restrict__ img, int rows, int cols, float centerX, float centerY, int blurRadius, float intensity);
__global__ void reverse_contrast(unsigned char* __restrict__ img, const int nPixels);
__global__ void reverseContrastRGBA_kernel(unsigned char* __restrict__ img, const int nPixels);
__global__ void shift_hue_kernel(unsigned char* __restrict__ img, const int nPixels, const float rotationFactor);
__global__ void shiftHueRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const float rotationFactor);
__global__ void outlines_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int shiftX, const int shiftY);
__global__ void outlinesRGBA_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int shiftX, const int shiftY);
__global__ void subtract_kernel(unsigned char* __restrict__ img1, const unsigned char* __restrict__ img2, const int nPixels);
__global__ void subtractRGBA_kernel(unsigned char* __restrict__ img1, const unsigned char* __restrict__ img2, const int nPixels);
__global__ void fastBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius);
__global__ void fastBlurRGBA_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius);
__global__ void trueBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius);
__global__ void monoChrome_kernel(unsigned char* __restrict__ img, const int nPixels);
__global__ void monoChromeRGBA_kernel(unsigned char* __restrict__ img, const int nPixels);
__global__ void passColors_kernel(unsigned char* __restrict__ img, const int size, const float* __restrict__ passThreshValues);
__global__ void passColorsRGBA_kernel(unsigned char* __restrict__ img, const int size, const float* __restrict__ passThreshValues);
__global__ void preciseBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius, const float precision);
__global__ void inverseColors_kernel(unsigned char* __restrict__ img, const int size);
__global__ void inverseColorsRGBA_kernel(unsigned char* __restrict__ img, const int size);
__global__ void blackNwhiteRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const float middle);
__global__ void blackNwhite_kernel(unsigned char* __restrict__ img, const int nPixels, const float middle);
__global__ void generateBinaryNoise(unsigned char* __restrict__ img, const int nPixels, size_t seed);
