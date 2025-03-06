#include "videoEffects.cuh"
#include <math_functions.h>


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

__global__ void pixelate_kernel(unsigned char* img, int rows, int cols, int pixelWidth, int pixelHeight) {
    // Calculate the block's starting position
    int blockX = blockIdx.x * pixelWidth;
    int blockY = blockIdx.y * pixelHeight;

    // Ensure the block is within the image bounds
    if (blockX >= cols || blockY >= rows) {
        return;
    }

    // Calculate the block's end position
    int blockEndX = min(blockX + pixelWidth, cols);
    int blockEndY = min(blockY + pixelHeight, rows);

    // Accumulate the sum of colors in the block
    size_t sum_colors[3] = { 0, 0, 0 };
    int pixelCount = 0;

    for (int y = blockY; y < blockEndY; ++y) {
        for (int x = blockX; x < blockEndX; ++x) {
            int idx = (y * cols + x) * 3;
            for (int c = 0; c < 3; ++c) {
                sum_colors[c] += img[idx + c];
            }
            pixelCount++;
        }
    }

    // Calculate the average color
    unsigned char avg_colors[3];
    for (int c = 0; c < 3; ++c) {
        avg_colors[c] = static_cast<unsigned char>(sum_colors[c] / pixelCount);
    }

    // Apply the average color to the entire block
    for (int y = blockY; y < blockEndY; ++y) {
        for (int x = blockX; x < blockEndX; ++x) {
            int idx = (y * cols + x) * 3;
            for (int c = 0; c < 3; ++c) {
                img[idx + c] = avg_colors[c];
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

__global__ void nearestColor_kernel(unsigned char* img, int rows, int cols, const unsigned char* colors_BGR, int num_colors) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int idx = (y * cols + x) * 3; // Index for the RGB channels

    // Get the current pixel's color
    unsigned char b = img[idx];
    unsigned char g = img[idx + 1];
    unsigned char r = img[idx + 2];

    // Initialize variables to find the nearest color
    int min_distance = INT_MAX;
    int nearest_color_idx = 0;

    // Iterate through the palette to find the nearest color
    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;

        // Get the palette color
        unsigned char pb = colors_BGR[palette_idx];
        unsigned char pg = colors_BGR[palette_idx + 1];
        unsigned char pr = colors_BGR[palette_idx + 2];

        // Calculate the squared Euclidean distance between the colors
        int db = b - pb;
        int dg = g - pg;
        int dr = r - pr;
        int distance = db * db + dg * dg + dr * dr;

        // Update the nearest color if this one is closer
        if (distance < min_distance) {
            min_distance = distance;
            nearest_color_idx = i;
        }
    }

    // Set the pixel to the nearest color
    int palette_idx = nearest_color_idx * 3;
    img[idx] = colors_BGR[palette_idx];     // Blue
    img[idx + 1] = colors_BGR[palette_idx + 1]; // Green
    img[idx + 2] = colors_BGR[palette_idx + 2]; // Red
}

__global__ void radial_blur_kernel(unsigned char* img, int rows, int cols, float centerX, float centerY, int blurRadius, float intensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the image bounds
    if (x >= cols || y >= rows) {
        return;
    }

    // Calculate the direction vector from the center to the current pixel
    float dirX = x - centerX;
    float dirY = y - centerY;

    // Normalize the direction vector
    float length = sqrtf(dirX * dirX + dirY * dirY);
    if (length > 0) {
        dirX /= length;
        dirY /= length;
    }

    // Accumulate color values along the radial direction
    float sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    for (int i = -blurRadius; i <= blurRadius; ++i) {
        int sampleX = x + static_cast<int>(dirX * i * intensity);
        int sampleY = y + static_cast<int>(dirY * i * intensity);

        // Clamp the sample coordinates to the image bounds
        sampleX = max(0, min(sampleX, cols - 1));
        sampleY = max(0, min(sampleY, rows - 1));

        // Get the color at the sampled pixel
        int idx = (sampleY * cols + sampleX) * 3;
        sumR += img[idx];
        sumG += img[idx + 1];
        sumB += img[idx + 2];
        count++;
    }

    // Calculate the average color
    unsigned char avgR = static_cast<unsigned char>(sumR / count);
    unsigned char avgG = static_cast<unsigned char>(sumG / count);
    unsigned char avgB = static_cast<unsigned char>(sumB / count);

    // Write the averaged color back to the image
    int idx = (y * cols + x) * 3;
    img[idx] = avgR;
    img[idx + 1] = avgG;
    img[idx + 2] = avgB;
}

__global__ void reverse_contrast(unsigned char* img, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the image bounds
    if (x >= cols || y >= rows) {
        return;
    }

    int idx = (y * cols + x) * 3;

    // Get the RGB values of the current pixel
    int r = img[idx];
    int g = img[idx+1];
    int b = img[idx+2];

    // Calculate the maximum and minimum RGB values
    unsigned char max_color = max(max(r, g), b);
    unsigned char min_color = min(min(r, g), b);

    // Calculate the "lightness" (average of max and min)
    unsigned char lightness = (max_color + min_color) / 2;

    // Invert the lightness
    unsigned char inverted_lightness = 255 - lightness;

    float scale = static_cast<float>(inverted_lightness) / lightness;

    // Adjust the RGB values based on the inverted lightness
    img[idx] = min(max(static_cast<int>(r * scale), 0), 255);
    img[idx + 1] = min(max(static_cast<int>(g * scale), 0), 255);
    img[idx + 2] = min(max(static_cast<int>(b * scale), 0), 255);
}

__device__ static inline void rgb_to_yiq(float r, float g, float b, float& y, float& i, float& q) {
    y = 0.299f * r + 0.587f * g + 0.114f * b;
    i = 0.596f * r - 0.274f * g - 0.322f * b;
    q = 0.211f * r - 0.523f * g + 0.312f * b;
}

__device__ static inline void yiq_to_rgb(float y, float i, float q, float& r, float& g, float& b) {
    r = y + 0.956f * i + 0.621f * q;
    g = y - 0.272f * i - 0.647f * q;
    b = y - 1.106f * i + 1.703f * q;
}

__global__ void shift_hue_kernel(unsigned char* img, int rows, int cols, float rotationFactor) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;

    if (bx >= cols || by >= rows) return;

    int idx = (by * cols + bx) * 3;
    float r = img[idx] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx + 2] / 255.0f;

    float y, i, q;
    rgb_to_yiq(r, g, b, y, i, q);

    float cos_theta, sin_theta;
    sincospif(rotationFactor, &sin_theta, &cos_theta);

    // Rotate in the I-Q plane
    float new_i = i * cos_theta - q * sin_theta;
    float new_q = i * sin_theta + q * cos_theta;

    yiq_to_rgb(y, new_i, new_q, r, g, b);

    img[idx] = static_cast<unsigned char>(min(max(r, 0.0f), 1.0f) * 255);
    img[idx + 1] = static_cast<unsigned char>(min(max(g, 0.0f), 1.0f) * 255);
    img[idx + 2] = static_cast<unsigned char>(min(max(b, 0.0f), 1.0f) * 255);
}