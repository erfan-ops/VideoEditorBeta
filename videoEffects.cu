#include "videoEffects.cuh"

__global__ void censor_kernel(unsigned char* img, int rows, int cols, int pixelWidth, int pixelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows || x % pixelWidth || y % pixelHeight) {
        return;
    }

    // Process every block of pixels
    // Calculate block boundaries
    int yw = (y + pixelHeight < rows) ? (y + pixelHeight) : rows;
    int xw = (x + pixelWidth < cols) ? (x + pixelWidth) : cols;

    int block_idx = (y * cols + x) * 3; // Top-left pixel index in the block

    // Apply censoring to all pixels in the block
    for (int i = y; i < yw; ++i) {
        for (int j = x; j < xw; ++j) {
            int idx = (i * cols + j) * 3; // RGB index for the pixel
            for (int c = 0; c < 3; ++c) {
                img[idx + c] = img[block_idx + c]; // Copy the color from the top-left pixel
            }
        }
    }
}

__global__ void roundColors_kernel(unsigned char* img, int rows, int cols, int thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int halfThresh = thresh / 2;

    int idx = (y * cols + x) * 3;
    int cidx;

    int colorValue;
    int result_value;

    for (int c = 0; c < 3; ++c) {
        cidx = idx + c;
        colorValue = img[cidx] + halfThresh;
        result_value = colorValue - (colorValue % thresh);
        img[cidx] = (result_value < 255) ? result_value : 255;
    }
}

__global__ void horizontalLine_kernel(unsigned char* img, int rows, int cols, int lineWidth, int thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows || (y % (lineWidth * 2)) >= lineWidth) {
        return;
    }

    int idx = (y * cols + x) * 3;
    int cidx;

    int result_value;

    for (int c = 0; c < 3; ++c) {
        cidx = idx + c;
        result_value = img[cidx] - thresh;
        img[cidx] = (result_value < 0) ? 0 : result_value;
    }
}

__global__ void triColor_kernel(unsigned char* img, int rows, int cols, const unsigned char* color1_BGR, const unsigned char* color2_BGR, const unsigned char* color3_BGR) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = (y * cols + x) * 3; // Index for the RGB channels
        float mediant = (img[idx] + img[idx + 1] + img[idx + 2]) / 765.0f; // Scale to 0-1
        float midpoint = 0.5f; // Midpoint of the gradient
        float scale_factor;

        if (mediant < midpoint) {
            // Blend from color1 to color2
            scale_factor = mediant / midpoint;
            for (int i = 0; i < 3; ++i) {
                img[idx + i] = static_cast<unsigned char>(
                    color1_BGR[i] + (color2_BGR[i] - color1_BGR[i]) * scale_factor);
            }
        }
        else {
            // Blend from color2 to color3
            scale_factor = (mediant - midpoint) / (1.0f - midpoint);
            for (int i = 0; i < 3; ++i) {
                img[idx + i] = static_cast<unsigned char>(
                    color2_BGR[i] + (color3_BGR[i] - color2_BGR[i]) * scale_factor);
            }
        }
    }
}


