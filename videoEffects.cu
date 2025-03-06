#include "videoEffects.cuh"


__global__ void censor_kernel(unsigned char* img, int rows, int cols, int pixelWidth, int pixelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int block_y = (y / pixelHeight) * pixelHeight;
    int block_x = (x / pixelWidth) * pixelWidth;

    int block_idx = (block_y * cols + block_x) * 3; // Top-left pixel index in the block
    int idx = (y * cols + x) * 3;

    for (int c = 0; c < 3; ++c) {
        img[idx + c] = img[block_idx + c]; // Copy the color from the top-left pixel
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

__global__ void dynamicColor_kernel(unsigned char* img, int rows, int cols, const unsigned char* colors_BGR, int num_colors) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int idx = (y * cols + x) * 3; // Index for the RGB channels
    float mediant = (img[idx] + img[idx + 1] + img[idx + 2]) / 765.0f; // Scale to 0-1

    // Calculate the segment of the gradient based on the number of colors
    float segment_size = 1.0f / (num_colors - 1);
    int segment_index = static_cast<int>(mediant / segment_size);
    segment_index = segment_index <= num_colors - 2 ? segment_index : num_colors - 2;

    // Calculate the blending factor within the segment
    float segment_start = segment_index * segment_size;
    float segment_end = (segment_index + 1) * segment_size;
    float scale_factor = (mediant - segment_start) / (segment_end - segment_start);

    // Blend the colors
    for (int i = 0; i < 3; ++i) {
        unsigned char color_start = colors_BGR[segment_index * 3 + i];
        unsigned char color_end = colors_BGR[(segment_index + 1) * 3 + i];
        img[idx + i] = static_cast<unsigned char>(
            color_start + (color_end - color_start) * scale_factor
            );
    }
}
