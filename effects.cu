#include "effects.cuh"
#include <math_functions.h>
#include <curand_kernel.h>
#include <cmath>


__global__ void censor_kernel(unsigned char* __restrict__ img, const int rows, const int cols, const int pixelWidth, const int pixelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int block_y = (y / pixelHeight) * pixelHeight;
    int block_x = (x / pixelWidth) * pixelWidth;

    int blockCenterIdx = ((block_y + pixelHeight / 2) * cols + (block_x + pixelWidth / 2)) * 3; // Centeral pixel index in the block
    int idx = (y * cols + x) * 3;

    for (int c = 0; c < 3; ++c) {
        img[idx + c] = img[blockCenterIdx + c];
    }
}

__global__ void censorRGBA_kernel(unsigned char* __restrict__ img, const int rows, const int cols, const int pixelWidth, const int pixelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int block_y = (y / pixelHeight) * pixelHeight;
    int block_x = (x / pixelWidth) * pixelWidth;

    int blockCenterIdx = ((block_y + pixelHeight / 2) * cols + (block_x + pixelWidth / 2)) * 4; // Top-left pixel index in the block
    int idx = (y * cols + x) * 4;

    for (int c = 0; c < 4; ++c) {
        img[idx + c] = img[blockCenterIdx + c]; // Copy the color from the top-left pixel
    }
}

__global__ void pixelate_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int blockY = (y / pixelHeight) * pixelHeight;
    int blockX = (x / pixelWidth) * pixelWidth;

    // Calculate the block's end position
    int blockEndX = min(blockX + pixelWidth, cols);
    int blockEndY = min(blockY + pixelHeight, rows);

    // Accumulate the sum of colors in the block
    size_t sum_colors[3] = { 0, 0, 0 };
    int pixelCount = 0;

    for (int y = blockY; y < blockEndY; ++y) {
        int yc = y * cols;
        for (int x = blockX; x < blockEndX; ++x) {
            int idx = (yc + x) * 3;
            for (int c = 0; c < 3; ++c) {
                sum_colors[c] += img[idx + c];
            }
            pixelCount++;
        }
    }

    // Calculate the average color
    unsigned char avg_colors[3]{ 0 };
    for (int c = 0; c < 3; ++c) {
        avg_colors[c] = static_cast<unsigned char>(sum_colors[c] / pixelCount);
    }

    // Apply the average color to the entire block
    for (int y = blockY; y < blockEndY; ++y) {
        int yc = y * cols;
        for (int x = blockX; x < blockEndX; ++x) {
            int idx = (yc + x) * 3;
            for (int c = 0; c < 3; ++c) {
                img[idx + c] = avg_colors[c];
            }
        }
    }
}

__global__ void pixelateRGBA_kernel(unsigned char* __restrict__ img, int rows, int cols, int pixelWidth, int pixelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int blockY = (y / pixelHeight) * pixelHeight;
    int blockX = (x / pixelWidth) * pixelWidth;

    // Calculate the block's end position
    int blockEndX = min(blockX + pixelWidth, cols);
    int blockEndY = min(blockY + pixelHeight, rows);

    // Accumulate the sum of colors in the block
    size_t sum_colors[4] = { 0, 0, 0, 0 };
    int pixelCount = 0;

    for (int y = blockY; y < blockEndY; ++y) {
        for (int x = blockX; x < blockEndX; ++x) {
            int idx = (y * cols + x) * 4;
            for (int c = 0; c < 4; ++c) {
                sum_colors[c] += img[idx + c];
            }
            pixelCount++;
        }
    }

    // Calculate the average color
    unsigned char avg_colors[4];
    for (int c = 0; c < 4; ++c) {
        avg_colors[c] = static_cast<unsigned char>(sum_colors[c] / pixelCount);
    }

    // Apply the average color to the entire block
    for (int y = blockY; y < blockEndY; ++y) {
        for (int x = blockX; x < blockEndX; ++x) {
            int idx = (y * cols + x) * 4;
            for (int c = 0; c < 4; ++c) {
                img[idx + c] = avg_colors[c];
            }
        }
    }
}

__global__ void roundColors_kernel(unsigned char* __restrict__ img, const int size, const float thresh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    float halfThresh = thresh / 2.0f;
    float colorValue = static_cast<float>(img[idx]) + halfThresh;
    float result_value = colorValue - fmodf(colorValue, thresh);

    img[idx] = static_cast<unsigned char>(fminf(result_value, 255.0f));
}

__global__ void roundColorsRGBA_kernel(unsigned char* __restrict__ img, const int size, const float thresh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 4 == 3 || idx >= size) return;

    float halfThresh = thresh / 2.0f;
    float colorValue = static_cast<float>(img[idx]) + halfThresh;
    float result_value = colorValue - fmodf(colorValue, thresh);

    img[idx] = static_cast<unsigned char>(fminf(result_value, 255.0f));
}

__global__ void horizontalLine_kernel(unsigned char* __restrict__ img, int rows, int cols, int lineWidth, int thresh) {
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

__global__ void dynamicColor_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ colors_BGR, const int num_colors) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3; // Index for the RGB channels
    float mediant = (0.114f * static_cast<float>(img[idx]) +
                     0.587f * static_cast<float>(img[idx + 1]) +
                     0.299f * static_cast<float>(img[idx + 2])) / 255.0f;

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

__global__ void dynamicColorRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ colors_BGR, const int num_colors) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 4; // Index for the RGB channels
    float mediant = (0.299f * static_cast<float>(img[idx]) +
        0.587f * static_cast<float>(img[idx + 1]) +
        0.114f * static_cast<float>(img[idx + 2])) / 255.0f;

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
        unsigned char color_start = colors_BGR[segment_index * 3 + (2 - i)];
        unsigned char color_end = colors_BGR[(segment_index + 1) * 3 + (2 - i)];
        img[idx + i] = static_cast<unsigned char>(
            color_start + (color_end - color_start) * scale_factor
        );
    }
}

__global__ void nearestColor_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ colors_BGR, const int num_colors) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

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

__global__ void nearestColorRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const unsigned char* __restrict__ color_BGR, const int num_colors) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    // Get the current pixel's color
    unsigned char r = img[idx];
    unsigned char g = img[idx + 1];
    unsigned char b = img[idx + 2];

    // Initialize variables to find the nearest color
    int min_distance = INT_MAX;
    int nearest_color_idx = 0;

    // Iterate through the palette to find the nearest color
    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;

        // Get the palette color
        unsigned char pr = color_BGR[palette_idx + 2];
        unsigned char pg = color_BGR[palette_idx + 1];
        unsigned char pb = color_BGR[palette_idx];

        // Calculate the squared Euclidean distance between the colors
        int dr = r - pr;
        int dg = g - pg;
        int db = b - pb;
        int distance = dr * dr + dg * dg + db * db;

        // Update the nearest color if this one is closer
        if (distance < min_distance) {
            min_distance = distance;
            nearest_color_idx = i;
        }
    }

    // Set the pixel to the nearest color
    int palette_idx = nearest_color_idx * 3;
    img[idx]     = color_BGR[palette_idx + 2]; // Red
    img[idx + 1] = color_BGR[palette_idx + 1]; // Green
    img[idx + 2] = color_BGR[palette_idx];     // Blue
}

__global__ void blendNearestColors_kernel(
    unsigned char* __restrict__ img,
    const int nPixels,
    const unsigned char* __restrict__ colors_BGR,
    const int num_colors
) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Get the current pixel's color
    unsigned char b = img[idx];
    unsigned char g = img[idx + 1];
    unsigned char r = img[idx + 2];

    // Temporary arrays to store weights and avoid recomputation
    float weights_sum = 0.0f;
    float color_sum_b = 0.0f;
    float color_sum_g = 0.0f;
    float color_sum_r = 0.0f;

    // Loop over all colors in the palette
    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;

        // Get palette color
        unsigned char pb = colors_BGR[palette_idx];
        unsigned char pg = colors_BGR[palette_idx + 1];
        unsigned char pr = colors_BGR[palette_idx + 2];

        // Calculate squared distance
        int db = int(b) - int(pb);
        int dg = int(g) - int(pg);
        int dr = int(r) - int(pr);
        float dist_squared = float(db * db + dg * dg + dr * dr);

        // Avoid division by zero: assign a very high weight to exact matches
        float weight = (dist_squared == 0.0f) ? 1e6f : 1.0f / dist_squared;

        weights_sum += weight;
        color_sum_b += weight * pb;
        color_sum_g += weight * pg;
        color_sum_r += weight * pr;
    }

    // Compute weighted average color
    img[idx] = static_cast<unsigned char>(fminf(fmaxf(color_sum_b / weights_sum, 0.0f), 255.0f));
    img[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(color_sum_g / weights_sum, 0.0f), 255.0f));
    img[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(color_sum_r / weights_sum, 0.0f), 255.0f));
}

__global__ void blendNearestColorsRGBA_kernel(
    unsigned char* __restrict__ img,
    const int nPixels,
    const unsigned char* __restrict__ colors_BGR,
    const int num_colors
) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    // Get the current pixel's color
    unsigned char r = img[idx];
    unsigned char g = img[idx + 1];
    unsigned char b = img[idx + 2];

    // Temporary arrays to store weights and avoid recomputation
    float weights_sum = 0.0f;
    float color_sum_b = 0.0f;
    float color_sum_g = 0.0f;
    float color_sum_r = 0.0f;

    // Loop over all colors in the palette
    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;

        // Get palette color
        unsigned char pb = colors_BGR[palette_idx];
        unsigned char pg = colors_BGR[palette_idx + 1];
        unsigned char pr = colors_BGR[palette_idx + 2];

        // Calculate squared distance
        int db = int(b) - int(pb);
        int dg = int(g) - int(pg);
        int dr = int(r) - int(pr);
        float dist_squared = float(db * db + dg * dg + dr * dr);

        // Avoid division by zero: assign a very high weight to exact matches
        float weight = (dist_squared == 0.0f) ? 1e6f : 1.0f / dist_squared;

        weights_sum += weight;
        color_sum_b += weight * pb;
        color_sum_g += weight * pg;
        color_sum_r += weight * pr;
    }

    // Compute weighted average color
    img[idx] = static_cast<unsigned char>(fminf(fmaxf(color_sum_r / weights_sum, 0.0f), 255.0f));
    img[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(color_sum_g / weights_sum, 0.0f), 255.0f));
    img[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(color_sum_b / weights_sum, 0.0f), 255.0f));
}

__global__ void radial_blur_kernel(unsigned char* __restrict__ img, int rows, int cols, float centerX, float centerY, int blurRadius, float intensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    float dirX = x - centerX;
    float dirY = y - centerY;

    float length = dirX * dirX + dirY * dirY;
    if (length > 0) {
        length = sqrtf(length);
        dirX /= length;
        dirY /= length;
    }

    float sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    for (int i = -blurRadius; i <= blurRadius; ++i) {
        int sampleX = x + static_cast<int>(dirX * i * intensity);
        int sampleY = y + static_cast<int>(dirY * i * intensity);

        sampleX = max(0, min(sampleX, cols - 1));
        sampleY = max(0, min(sampleY, rows - 1));

        int idx = (sampleY * cols + sampleX) * 3;
        sumR += img[idx];
        sumG += img[idx + 1];
        sumB += img[idx + 2];
        count++;
    }

    unsigned char avgR = static_cast<unsigned char>(sumR / count);
    unsigned char avgG = static_cast<unsigned char>(sumG / count);
    unsigned char avgB = static_cast<unsigned char>(sumB / count);

    int idx = (y * cols + x) * 3;
    img[idx] = avgR;
    img[idx + 1] = avgG;
    img[idx + 2] = avgB;
}

__global__ void radialBlurRGBA_kernel(unsigned char* __restrict__ img, int rows, int cols, float centerX, float centerY, int blurRadius, float intensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    float dirX = x - centerX;
    float dirY = y - centerY;

    float length = dirX * dirX + dirY * dirY;
    if (length > 0) {
        length = sqrtf(length);
        dirX /= length;
        dirY /= length;
    }

    float sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    for (int i = -blurRadius; i <= blurRadius; ++i) {
        int sampleX = x + static_cast<int>(dirX * i * intensity);
        int sampleY = y + static_cast<int>(dirY * i * intensity);

        sampleX = max(0, min(sampleX, cols - 1));
        sampleY = max(0, min(sampleY, rows - 1));

        int idx = (sampleY * cols + sampleX) * 4;
        sumR += img[idx++];
        sumG += img[idx++];
        sumB += img[idx];
        count++;
    }

    unsigned char avgR = static_cast<unsigned char>(sumR / count);
    unsigned char avgG = static_cast<unsigned char>(sumG / count);
    unsigned char avgB = static_cast<unsigned char>(sumB / count);

    int idx = (y * cols + x) * 4;
    img[idx++] = avgR;
    img[idx++] = avgG;
    img[idx]   = avgB;
}

__global__ void reverse_contrast(unsigned char* __restrict__ img, const int nPixels) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Load RGB components and normalize to [0,1]
    float b = img[idx] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float r = img[idx + 2] / 255.0f;

    // Compute max and min values
    float max_color = fmaxf(fmaxf(r, g), b);
    float min_color = fminf(fminf(r, g), b);

    // Compute original lightness (L)
    float l = 0.5f * (max_color + min_color);

    // Invert lightness
    float inverted_l = 1.0f - l;

    // Avoid division by zero
    float delta = max_color - min_color;
    if (delta < 1e-6f) {
        // If the color is grayscale, simply invert it
        img[idx] = static_cast<unsigned char>(inverted_l * 255.0f);
        img[idx + 1] = static_cast<unsigned char>(inverted_l * 255.0f);
        img[idx + 2] = static_cast<unsigned char>(inverted_l * 255.0f);
        return;
    }

    // Compute saturation (S) to maintain color intensity
    float s = delta / (1.0f - fabsf(2.0f * l - 1.0f));

    // Compute new min/max values based on inverted lightness
    float new_max = inverted_l + s * (1.0f - fabsf(2.0f * inverted_l - 1.0f)) * 0.5f;
    float new_min = inverted_l - s * (1.0f - fabsf(2.0f * inverted_l - 1.0f)) * 0.5f;

    // Remap RGB values while preserving hue
    float new_r = (r == max_color) ? new_max : new_min + (r - min_color) * (new_max - new_min) / delta;
    float new_g = (g == max_color) ? new_max : new_min + (g - min_color) * (new_max - new_min) / delta;
    float new_b = (b == max_color) ? new_max : new_min + (b - min_color) * (new_max - new_min) / delta;

    // Store back to image buffer
    img[idx] = static_cast<unsigned char>(fminf(fmaxf(new_b * 255.0f, 0.0f), 255.0f));
    img[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(new_g * 255.0f, 0.0f), 255.0f));
    img[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(new_r * 255.0f, 0.0f), 255.0f));
}

__global__ void reverseContrastRGBA_kernel(unsigned char* __restrict__ img, const int nPixels) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    // Load RGB components and normalize to [0,1]
    float r = img[idx] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx + 2] / 255.0f;

    // Compute max and min values
    float max_color = fmaxf(fmaxf(r, g), b);
    float min_color = fminf(fminf(r, g), b);

    // Compute original lightness (L)
    float l = 0.5f * (max_color + min_color);

    // Invert lightness
    float inverted_l = 1.0f - l;

    // Avoid division by zero
    float delta = max_color - min_color;
    if (delta < 1e-6f) {
        // If the color is grayscale, simply invert it
        img[idx] = static_cast<unsigned char>(inverted_l * 255.0f);
        img[idx + 1] = static_cast<unsigned char>(inverted_l * 255.0f);
        img[idx + 2] = static_cast<unsigned char>(inverted_l * 255.0f);
        return;
    }

    // Compute saturation (S) to maintain color intensity
    float denominator = fmaf(2.0f, l, -1.0f); // 2.0f * l - 1.0f
    denominator = 1.0f - fabsf(denominator);
    float s = delta / denominator;

    // Compute new min/max values based on inverted lightness
    float term = fmaf(2.0f, inverted_l, -1.0f); // 2.0f * inverted_l - 1.0f
    term = 1.0f - fabsf(term);
    term = fmaf(s, term, 0.0f) * 0.5f; // s * term * 0.5f
    float new_max = inverted_l + term;
    float new_min = inverted_l - term;

    // Remap RGB values while preserving hue
    float ratio = (new_max - new_min) / delta;
    float new_r = (r == max_color) ? new_max : fmaf(r - min_color, ratio, new_min);
    float new_g = (g == max_color) ? new_max : fmaf(g - min_color, ratio, new_min);
    float new_b = (b == max_color) ? new_max : fmaf(b - min_color, ratio, new_min);

    // Store back to image buffer
    img[idx] = static_cast<unsigned char>(fminf(fmaxf(new_r * 255.0f, 0.0f), 255.0f));
    img[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(new_g * 255.0f, 0.0f), 255.0f));
    img[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(new_b * 255.0f, 0.0f), 255.0f));
}

__device__ static __inline__ void rgb_to_yiq(float r, float g, float b, float& y, float& i, float& q) {
    y = 0.299f * r + 0.587f * g + 0.114f * b;
    i = 0.596f * r - 0.274f * g - 0.322f * b;
    q = 0.211f * r - 0.523f * g + 0.312f * b;
}

__device__ static __inline__ void yiq_to_rgb(float y, float i, float q, float& r, float& g, float& b) {
    r = y + 0.956f * i + 0.621f * q;
    g = y - 0.272f * i - 0.647f * q;
    b = y - 1.106f * i + 1.703f * q;
}

__global__ void shift_hue_kernel(unsigned char* __restrict__ img, const int nPixels, const float rotationFactor) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    float b = img[idx] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float r = img[idx + 2] / 255.0f;

    float y, i, q;
    rgb_to_yiq(r, g, b, y, i, q);

    float cos_theta, sin_theta;
    sincospif(rotationFactor, &sin_theta, &cos_theta);

    // Rotate in the I-Q plane
    float new_i = i * cos_theta - q * sin_theta;
    float new_q = i * sin_theta + q * cos_theta;

    yiq_to_rgb(y, new_i, new_q, r, g, b);

    img[idx] = static_cast<unsigned char>(min(max(b, 0.0f), 1.0f) * 255);
    img[idx + 1] = static_cast<unsigned char>(min(max(g, 0.0f), 1.0f) * 255);
    img[idx + 2] = static_cast<unsigned char>(min(max(r, 0.0f), 1.0f) * 255);
}

__global__ void shiftHueRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const float rotationFactor) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

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

__global__ void outlines_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int shiftX, const int shiftY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = (y * cols + x) * 3;

    if (x < cols - shiftX && y < rows - shiftY) {
        int shiftedIdx = idx + 3 * (shiftY * cols + shiftX);

        for (int c = 0; c < 3; c++) {
            int color_idx = idx + c;
            img[color_idx] = static_cast<unsigned char>(abs(static_cast<short>(img_copy[color_idx]) - img_copy[shiftedIdx + c]));
        }
    }
    else {
        img[idx++] = 0;
        img[idx++] = 0;
        img[idx] = 0;
    }
}

__global__ void outlinesRGBA_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int shiftX, const int shiftY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = (y * cols + x) * 4;

    if (x < cols - shiftX && y < rows - shiftY) {
        int shiftedIdx = idx + 4 * (shiftY * cols + shiftX);

        for (int c = 0; c < 3; c++) {
            int color_idx = idx + c;
            img[color_idx] = static_cast<unsigned char>(abs(static_cast<short>(img_copy[color_idx]) - img_copy[shiftedIdx + c]));
        }
    }
    else {
        img[idx++] = 0;
        img[idx++] = 0;
        img[idx] = 0;
    }
}

__global__ void subtract_kernel(unsigned char* __restrict__ img1, const unsigned char* __restrict__ img2, const int nPixels) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    for (int c = 0; c < 3; c++) {
        int color_idx = idx + c;
        short diff = static_cast<short>(img1[color_idx]) - static_cast<short>(img2[color_idx]);
        img1[color_idx] = static_cast<unsigned char>(abs(diff));
    }
}

__global__ void subtractRGBA_kernel(unsigned char* __restrict__ img1, const unsigned char* __restrict__ img2, const int nPixels) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    for (int c = 0; c < 3; c++) {
        int color_idx = idx + c;
        short diff = static_cast<short>(img1[color_idx]) - static_cast<short>(img2[color_idx]);
        img1[color_idx] = static_cast<unsigned char>(abs(diff));
    }
}

__global__ void fastBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the image bounds
    if (x >= cols || y >= rows) {
        return;
    }

    int idx = (y * cols + x) * 3;
    int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    const int blur_radius_sqr = blur_radius * blur_radius;

    // Precalculate boundaries
    int startX = max(0, x - blur_radius);
    int endX = min(cols - 1, x + blur_radius);
    int startY = max(0, y - blur_radius);
    int endY = min(rows - 1, y + blur_radius);

    for (int sampleX = startX; sampleX <= endX; ++sampleX) {
        int i = sampleX - x;  // Calculate relative position

        for (int sampleY = startY; sampleY <= endY; ++sampleY) {
            int j = sampleY - y;  // Calculate relative position

            // Calculate the distance from the center pixel
            float distance_sqr = i * i + j * j;
            if (distance_sqr <= blur_radius_sqr) {
                int sampleIdx = (sampleY * cols + sampleX) * 3;
                sumR += img_copy[sampleIdx];
                sumG += img_copy[sampleIdx + 1];
                sumB += img_copy[sampleIdx + 2];
                count++;
            }
        }
    }

    // Write the averaged color back to the original image
    img[idx] = static_cast<unsigned char>(sumR / count);
    img[idx + 1] = static_cast<unsigned char>(sumG / count);
    img[idx + 2] = static_cast<unsigned char>(sumB / count);
}

__global__ void fastBlurRGBA_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the image bounds
    if (x >= cols || y >= rows) {
        return;
    }

    int idx = (y * cols + x) * 4;
    int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    const int blur_radius_sqr = blur_radius * blur_radius;

    // Precalculate boundaries
    int startX = max(0, x - blur_radius);
    int endX = min(cols - 1, x + blur_radius);
    int startY = max(0, y - blur_radius);
    int endY = min(rows - 1, y + blur_radius);

    for (int sampleY = startY; sampleY <= endY; ++sampleY) {
        int j = sampleY - y;  // Calculate relative position
        int helper = sampleY * cols;
        for (int sampleX = startX; sampleX <= endX; ++sampleX) {
            int i = sampleX - x;  // Calculate relative position

            // Calculate the distance from the center pixel
            float distance_sqr = i * i + j * j;
            if (distance_sqr <= blur_radius_sqr) {
                int sampleIdx = (helper + sampleX) * 4;
                sumR += img_copy[sampleIdx];
                sumG += img_copy[sampleIdx + 1];
                sumB += img_copy[sampleIdx + 2];
                count++;
            }
        }
    }

    // Write the averaged color back to the original image
    img[idx] = static_cast<unsigned char>(sumR / count);
    img[idx + 1] = static_cast<unsigned char>(sumG / count);
    img[idx + 2] = static_cast<unsigned char>(sumB / count);
}

__global__ void trueBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the image bounds
    if (x >= cols || y >= rows) {
        return;
    }

    int idx = (y * cols + x) * 3;

    float sumR = 0, sumG = 0, sumB = 0;
    float totalWeight = 0;

    int blur_radius_sqr = blur_radius * blur_radius;

    // Iterate over the rounded neighborhood
    for (int i = -blur_radius - 1; i <= blur_radius + 1; ++i) {
        for (int j = -blur_radius - 1; j <= blur_radius + 1; ++j) {
            int sampleX = x + i;
            int sampleY = y + j;

            // Check if the sampled pixel is within the image bounds
            if (sampleX >= 0 && sampleX < cols && sampleY >= 0 && sampleY < rows) {
                int sampleIdx = (sampleY * cols + sampleX) * 3;

                // Calculate the distance from the center pixel
                float distance_sqr = i * i + j * j;

                // Calculate the blending factor
                float weight = 1.0f;
                if (distance_sqr > blur_radius_sqr) {
                    weight = (blur_radius + 1) - sqrtf(distance_sqr); // Smooth transition beyond the blur radius
                    if (weight < 0) weight = 0; // Clamp to 0 for pixels too far away
                }

                // Accumulate the weighted color values
                sumR += img_copy[sampleIdx] * weight;
                sumG += img_copy[sampleIdx + 1] * weight;
                sumB += img_copy[sampleIdx + 2] * weight;
                totalWeight += weight;
            }
        }
    }

    // Normalize the accumulated color values by the total weight
    if (totalWeight > 0) {
        img[idx] = static_cast<unsigned char>(sumR / totalWeight); // Red
        img[idx + 1] = static_cast<unsigned char>(sumG / totalWeight); // Green
        img[idx + 2] = static_cast<unsigned char>(sumB / totalWeight); // Blue
    }
}

__global__ void monoChrome_kernel(unsigned char* __restrict__ img, const int nPixels) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    unsigned char m = static_cast<unsigned char>(0.114f * static_cast<float>(img[idx]) +
                                                 0.587f * static_cast<float>(img[idx + 1]) +
                                                 0.299f * static_cast<float>(img[idx + 2]));

    img[idx++] = m;
    img[idx++] = m;
    img[idx] = m;
}

__global__ void monoChromeRGBA_kernel(unsigned char* __restrict__ img, const int nPixels) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    unsigned char m = static_cast<unsigned char>(0.299f * static_cast<float>(img[idx]) +
                                                 0.587f * static_cast<float>(img[idx + 1]) +
                                                 0.114f * static_cast<float>(img[idx + 2]));

    img[idx++] = m;
    img[idx++] = m;
    img[idx] = m;
}

__global__ void passColors_kernel(unsigned char* __restrict__ img, const int size, const float* __restrict__ passThreshValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    img[idx] = passThreshValues[idx % 3] * img[idx];
}

__global__ void passColorsRGBA_kernel(unsigned char* __restrict__ img, const int size, const float* __restrict__ passThreshValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    img[idx] = passThreshValues[idx % 4] * img[idx];
}

__device__ static __inline__ float calculatePixelWeight(const float x, const float y, const float cx, const float cy, const float r, const float precision) {
    // Define the pixel boundaries
    float x0 = x - 0.5f; // Left edge of the pixel
    float x1 = x + 0.5f; // Right edge of the pixel
    float y0 = y - 0.5f; // Bottom edge of the pixel
    float y1 = y + 0.5f; // Top edge of the pixel

    // Clamp the boundaries to the circle
    x0 = fmaxf(x0, cx - r);
    x1 = fminf(x1, cx + r);
    y0 = fmaxf(y0, cy - r);
    y1 = fminf(y1, cy + r);

    // Calculate the weight of the pixel based on its overlap with the circle
    float weight = 0.0f;
    for (float px = x0; px <= x1; px += precision) {
        for (float py = y0; py <= y1; py += precision) {
            float dx = px - cx;
            float dy = py - cy;
            if (dx * dx + dy * dy <= r * r) {
                weight += precision * precision; // Add the weight of the small square
            }
        }
    }

    return weight;
}

__global__ void preciseBlur_kernel(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius, const float precision) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    // Initialize sums for each color channel
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    float totalWeight = 0.0f;

    // Iterate over the circular neighborhood
    for (int i = -blur_radius; i <= blur_radius; ++i) {
        for (int j = -blur_radius; j <= blur_radius; ++j) {
            int sampleX = x + i;
            int sampleY = y + j;

            // Check if the sampled pixel is within the image bounds
            if (sampleX >= 0 && sampleX < cols && sampleY >= 0 && sampleY < rows) {
                // Calculate the weight of the sampled pixel
                float weight = calculatePixelWeight(sampleX, sampleY, x, y, blur_radius, precision);

                // Accumulate the weighted color values
                int sampleIdx = (sampleY * cols + sampleX) * 3;
                sumR += img_copy[sampleIdx] * weight;
                sumG += img_copy[sampleIdx + 1] * weight;
                sumB += img_copy[sampleIdx + 2] * weight;
                totalWeight += weight;
            }
        }
    }

    // Normalize the accumulated color values by the total weight
    if (totalWeight > 0.0f) {
        int idx = (y * cols + x) * 3;
        img[idx] = static_cast<unsigned char>(sumR / totalWeight); // Red
        img[idx + 1] = static_cast<unsigned char>(sumG / totalWeight); // Green
        img[idx + 2] = static_cast<unsigned char>(sumB / totalWeight); // Blue
    }
}

__global__ void inverseColors_kernel(unsigned char* __restrict__ img, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    img[idx] = 255ui8 - img[idx];
}

__global__ void inverseColorsRGBA_kernel(unsigned char* __restrict__ img, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 4 == 3 || idx >= size) return;
    img[idx] = 255ui8 - img[idx];
}

__global__ void blackNwhite_kernel(unsigned char* __restrict__ img, const int nPixels, const float middle) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    float m = 0.114f * static_cast<float>(img[idx]) +
              0.587f * static_cast<float>(img[idx + 1]) +
              0.299f * static_cast<float>(img[idx + 2]);

    unsigned char c = m > middle ? 255 : 0;

    img[idx++] = c;
    img[idx++] = c;
    img[idx] = c;
}

__global__ void blackNwhiteRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const float middle) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    float m = (0.299f * static_cast<float>(img[idx]) +
        0.587f * static_cast<float>(img[idx + 1]) +
        0.114f * static_cast<float>(img[idx + 2])) * (static_cast<float>(img[idx + 3]) / 255.0f);

    unsigned char c = m > middle ? 255 : 0;

    img[idx++] = c;
    img[idx++] = c;
    img[idx] = c;
}

__global__ void generateBinaryNoise(unsigned char* __restrict__ img, const int nPixels, size_t seed) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Use a more unique seed by incorporating blockIdx.x and threadIdx.x
    curandState state;
    curand_init(seed, pIdx, 0, &state);

    // Generate a random binary value (0 or 255)
    unsigned char c = (curand(&state) & 1) * 255;

    // Set all three channels (RGB) to the same value
    img[idx++] = c;
    img[idx++] = c;
    img[idx] = c;
}

__device__ static __inline__ void RGBtoHSL(float r, float g, float b, float& h, float& s, float& l) {
    float max_c = fmaxf(fmaxf(r, g), b);
    float min_c = fminf(fminf(r, g), b);
    float delta = max_c - min_c;

    l = 0.5f * (max_c + min_c);

    if (delta < 1e-6f) {
        h = 0.0f;
        s = 0.0f;
        return;
    }

    s = delta / (1.0f - fabsf(2.0f * l - 1.0f));

    if (max_c == r)
        h = fmodf((g - b) / delta, 6.0f);
    else if (max_c == g)
        h = ((b - r) / delta) + 2.0f;
    else
        h = ((r - g) / delta) + 4.0f;

    h /= 6.0f;
    if (h < 0.0f) h += 1.0f;
}

__device__ static __inline__ float hueToRGB(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

__device__ static __inline__ void HSLtoRGB(float h, float s, float l, float& r, float& g, float& b) {
    if (s < 1e-6f) {
        r = g = b = l;
        return;
    }

    float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
    float p = 2.0f * l - q;
    r = hueToRGB(p, q, h + 1.0f / 3.0f);
    g = hueToRGB(p, q, h);
    b = hueToRGB(p, q, h - 1.0f / 3.0f);
}

__global__ void fixedLightness_kernel(unsigned char* __restrict__ img, const int nPixels, const float lightness) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Read and normalize RGB
    float r = img[idx + 2] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx] / 255.0f;

    // Convert to HSL
    float h, s, l;
    RGBtoHSL(r, g, b, h, s, l);

    // Set lightness
    l = lightness;

    // Convert back to RGB
    HSLtoRGB(h, s, l, r, g, b);

    // Write back
    img[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
    img[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
    img[idx] = static_cast<unsigned char>(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
}

__global__ void fixedLightnessRGBA_kernel(unsigned char* __restrict__ img, const int nPixels, const float lightness) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    // Read and normalize RGB
    float r = img[idx + 2] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx] / 255.0f;

    // Convert to HSL
    float h, s, l;
    RGBtoHSL(r, g, b, h, s, l);

    // Set lightness
    l = lightness;

    // Convert back to RGB
    HSLtoRGB(h, s, l, r, g, b);

    // Write back
    img[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
    img[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
    img[idx] = static_cast<unsigned char>(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
}
