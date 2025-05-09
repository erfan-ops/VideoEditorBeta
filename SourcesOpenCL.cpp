#include "SourcesOpenCL.h"

const char* softPaletteOpenCLKernelSource = R"CLC(
__kernel void blendNearestColors_kernel(
    __global uchar* img,
    const int nPixels,
    __global const uchar* colors_BGR,
    const int num_colors
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    uchar b = img[idx];
    uchar g = img[idx + 1];
    uchar r = img[idx + 2];

    float weights_sum = 0.0f;
    float color_sum_b = 0.0f;
    float color_sum_g = 0.0f;
    float color_sum_r = 0.0f;

    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;
        uchar pb = colors_BGR[palette_idx];
        uchar pg = colors_BGR[palette_idx + 1];
        uchar pr = colors_BGR[palette_idx + 2];

        int db = (int)b - (int)pb;
        int dg = (int)g - (int)pg;
        int dr = (int)r - (int)pr;

        float dist_squared = (float)(db * db + dg * dg + dr * dr);
        float weight = (dist_squared == 0.0f) ? 1e6f : 1.0f / dist_squared;

        weights_sum += weight;
        color_sum_b += weight * (float)pb;
        color_sum_g += weight * (float)pg;
        color_sum_r += weight * (float)pr;
    }

    img[idx]     = (uchar)clamp(color_sum_b / weights_sum, 0.0f, 255.0f);
    img[idx + 1] = (uchar)clamp(color_sum_g / weights_sum, 0.0f, 255.0f);
    img[idx + 2] = (uchar)clamp(color_sum_r / weights_sum, 0.0f, 255.0f);
}
)CLC";

const char* softPaletteRGBAOpenCLKernelSource = R"CLC(
__kernel void blendNearestColorsRGBA_kernel(
    __global uchar* img,
    const int nPixels,
    __global const uchar* colors_BGR,
    const int num_colors
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    uchar r = img[idx];
    uchar g = img[idx + 1];
    uchar b = img[idx + 2];

    float weights_sum = 0.0f;
    float color_sum_b = 0.0f;
    float color_sum_g = 0.0f;
    float color_sum_r = 0.0f;

    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;
        uchar pb = colors_BGR[palette_idx];
        uchar pg = colors_BGR[palette_idx + 1];
        uchar pr = colors_BGR[palette_idx + 2];

        int db = (int)b - (int)pb;
        int dg = (int)g - (int)pg;
        int dr = (int)r - (int)pr;

        float dist_squared = (float)(db * db + dg * dg + dr * dr);
        float weight = (dist_squared == 0.0f) ? 1e6f : 1.0f / dist_squared;

        weights_sum += weight;
        color_sum_b += weight * (float)pb;
        color_sum_g += weight * (float)pg;
        color_sum_r += weight * (float)pr;
    }

    img[idx]     = (uchar)clamp(color_sum_r / weights_sum, 0.0f, 255.0f);
    img[idx + 1] = (uchar)clamp(color_sum_g / weights_sum, 0.0f, 255.0f);
    img[idx + 2] = (uchar)clamp(color_sum_b / weights_sum, 0.0f, 255.0f);
}
)CLC";

const char* blackAndWhiteOpenCLKernelSource = R"CLC(
__kernel void blackAndWhite_kernel(
    __global uchar* img,
    const int nPixels,
    const float middle
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    float m = 0.114f * (float)(img[idx]) +
        0.587f * (float)(img[idx + 1]) +
        0.299f * (float)(img[idx + 2]);

    unsigned char c = m > middle ? 255 : 0;

    img[idx++] = c;
    img[idx++] = c;
    img[idx] = c;
}
)CLC";

const char* blackAndWhiteRGBAOpenCLKernelSource = R"CLC(
__kernel void blackAndWhiteRGBA_kernel(
    __global uchar* img,
    const int nPixels,
    const float middle
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    float m = 0.114f * (float)(img[idx]) +
        0.587f * (float)(img[idx + 1]) +
        0.299f * (float)(img[idx + 2]);

    unsigned char c = m > middle ? 255 : 0;

    img[idx++] = c;
    img[idx++] = c;
    img[idx] = c;
}
)CLC";

const char* blurOpenCLKernelSource = R"CLC(
__kernel void blur_kernel(
    __global uchar* img,
    __global const uchar* img_copy,
    const int rows,
    const int cols,
    const int blur_radius
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    int idx = (y * cols + x) * 3;
    int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    int blur_radius_sqr = blur_radius * blur_radius;

    int startX = max(0, x - blur_radius);
    int endX = min(cols - 1, x + blur_radius);
    int startY = max(0, y - blur_radius);
    int endY = min(rows - 1, y + blur_radius);

    for (int sampleX = startX; sampleX <= endX; ++sampleX) {
        int i = sampleX - x;
        for (int sampleY = startY; sampleY <= endY; ++sampleY) {
            int j = sampleY - y;
            float distance_sqr = (float)(i * i + j * j);

            if (distance_sqr <= blur_radius_sqr) {
                int sampleIdx = (sampleY * cols + sampleX) * 3;
                sumR += img_copy[sampleIdx];
                sumG += img_copy[sampleIdx + 1];
                sumB += img_copy[sampleIdx + 2];
                count++;
            }
        }
    }

    img[idx]     = (uchar)(sumR / count);
    img[idx + 1] = (uchar)(sumG / count);
    img[idx + 2] = (uchar)(sumB / count);
}
)CLC";

const char* blurRGBAOpenCLKernelSource = R"CLC(
__kernel void blurRGBA_kernel(
    __global uchar* img,
    __global const uchar* img_copy,
    const int rows,
    const int cols,
    const int blur_radius
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    int idx = (y * cols + x) * 4;
    int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    int blur_radius_sqr = blur_radius * blur_radius;

    int startX = max(0, x - blur_radius);
    int endX = min(cols - 1, x + blur_radius);
    int startY = max(0, y - blur_radius);
    int endY = min(rows - 1, y + blur_radius);

    for (int sampleX = startX; sampleX <= endX; ++sampleX) {
        int i = sampleX - x;
        for (int sampleY = startY; sampleY <= endY; ++sampleY) {
            int j = sampleY - y;
            float distance_sqr = (float)(i * i + j * j);

            if (distance_sqr <= blur_radius_sqr) {
                int sampleIdx = (sampleY * cols + sampleX) * 4;
                sumR += img_copy[sampleIdx];
                sumG += img_copy[sampleIdx + 1];
                sumB += img_copy[sampleIdx + 2];
                count++;
            }
        }
    }

    img[idx]     = (uchar)(sumR / count);
    img[idx + 1] = (uchar)(sumG / count);
    img[idx + 2] = (uchar)(sumB / count);
}
)CLC";

const char* censorOpenCLKernelSource = R"CLC(
__kernel void censor_kernel(
    __global uchar* img,
    const int rows,
    const int cols,
    const int pixelWidth,
    const int pixelHeight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    int block_y = (y / pixelHeight) * pixelHeight;
    int block_x = (x / pixelWidth) * pixelWidth;

    int blockCenterIdx = ((block_y + pixelHeight / 2) * cols + (block_x + pixelWidth / 2)) * 3; // Centeral pixel index in the block
    int idx = (y * cols + x) * 3;

    for (int c = 0; c < 3; ++c) {
        img[idx + c] = img[blockCenterIdx + c]; // Copy the color from the top-left pixel
    }
}
)CLC";

const char* censorRGBAOpenCLKernelSource = R"CLC(
__kernel void censorRGBA_kernel(
    __global uchar* img,
    const int rows,
    const int cols,
    const int pixelWidth,
    const int pixelHeight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    int block_y = (y / pixelHeight) * pixelHeight;
    int block_x = (x / pixelWidth) * pixelWidth;

    int blockCenterIdx = ((block_y + pixelHeight / 2) * cols + (block_x + pixelWidth / 2)) * 4; // Centeral pixel index in the block
    int idx = (y * cols + x) * 4;

    for (int c = 0; c < 3; ++c) {
        img[idx + c] = img[blockCenterIdx + c];
    }
}
)CLC";

const char* flatLightOpenCLKernelSource = R"CLC(
inline float hueToRGB(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

__kernel void flatLight_kernel(
    __global uchar* img,
    const int nPixels,
    const float lightness
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Read and normalize RGB (assuming BGR layout like OpenCV)
    float r = img[idx + 2] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx + 0] / 255.0f;

    // RGB to HSL
    float max_c = fmax(fmax(r, g), b);
    float min_c = fmin(fmin(r, g), b);
    float delta = max_c - min_c;

    float h = 0.0f;
    float s = 0.0f;
    float l = 0.5f * (max_c + min_c);

    if (delta > 1e-6f) {
        s = delta / (1.0f - fabs(2.0f * l - 1.0f));

        if (max_c == r)
            h = fmod((g - b) / delta, 6.0f);
        else if (max_c == g)
            h = ((b - r) / delta) + 2.0f;
        else
            h = ((r - g) / delta) + 4.0f;

        h /= 6.0f;
        if (h < 0.0f) h += 1.0f;
    }

    // Set fixed lightness
    l = lightness;

    // HSL to RGB
    float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
    float p = 2.0f * l - q;

    float t_r = h + 1.0f / 3.0f;
    float t_g = h;
    float t_b = h - 1.0f / 3.0f;

    float r_out, g_out, b_out;

    r_out = hueToRGB(p, q, t_r);
    g_out = hueToRGB(p, q, t_g);
    b_out = hueToRGB(p, q, t_b);

    // Store back (still assuming BGR)
    img[idx + 2] = (uchar)(clamp(r_out * 255.0f, 0.0f, 255.0f));
    img[idx + 1] = (uchar)(clamp(g_out * 255.0f, 0.0f, 255.0f));
    img[idx + 0] = (uchar)(clamp(b_out * 255.0f, 0.0f, 255.0f));
}
)CLC";

const char* flatLightRGBAOpenCLKernelSource = R"CLC(
inline float hueToRGB(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

__kernel void flatLightRGBA_kernel(
    __global uchar* img,
    const int nPixels,
    const float lightness
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    // Read and normalize RGB (assuming BGR layout like OpenCV)
    float r = img[idx + 2] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx + 0] / 255.0f;

    // RGB to HSL
    float max_c = fmax(fmax(r, g), b);
    float min_c = fmin(fmin(r, g), b);
    float delta = max_c - min_c;

    float h = 0.0f;
    float s = 0.0f;
    float l = 0.5f * (max_c + min_c);

    if (delta > 1e-6f) {
        s = delta / (1.0f - fabs(2.0f * l - 1.0f));

        if (max_c == r)
            h = fmod((g - b) / delta, 6.0f);
        else if (max_c == g)
            h = ((b - r) / delta) + 2.0f;
        else
            h = ((r - g) / delta) + 4.0f;

        h /= 6.0f;
        if (h < 0.0f) h += 1.0f;
    }

    // Set fixed lightness
    l = lightness;

    // HSL to RGB
    float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
    float p = 2.0f * l - q;

    float t_r = h + 1.0f / 3.0f;
    float t_g = h;
    float t_b = h - 1.0f / 3.0f;

    float r_out, g_out, b_out;

    r_out = hueToRGB(p, q, t_r);
    g_out = hueToRGB(p, q, t_g);
    b_out = hueToRGB(p, q, t_b);

    // Store back (still assuming BGR)
    img[idx + 2] = (uchar)(clamp(r_out * 255.0f, 0.0f, 255.0f));
    img[idx + 1] = (uchar)(clamp(g_out * 255.0f, 0.0f, 255.0f));
    img[idx + 0] = (uchar)(clamp(b_out * 255.0f, 0.0f, 255.0f));
}
)CLC";

const char* changePaletteOpenCLKernelSource = R"CLC(
__kernel void changePalette_kernel(
    __global uchar* img,
    const int nPixels,
    __global const uchar* colors_BGR,
    const int num_colors
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    uchar b = img[idx];
    uchar g = img[idx + 1];
    uchar r = img[idx + 2];

    int min_distance = INT_MAX;
    int nearest_color_idx = 0;

    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;

        uchar pb = colors_BGR[palette_idx];
        uchar pg = colors_BGR[palette_idx + 1];
        uchar pr = colors_BGR[palette_idx + 2];

        int db = (int)b - (int)pb;
        int dg = (int)g - (int)pg;
        int dr = (int)r - (int)pr;

        int distance = db * db + dg * dg + dr * dr;

        if (distance < min_distance) {
            min_distance = distance;
            nearest_color_idx = i;
        }
    }

    int palette_idx = nearest_color_idx * 3;
    img[idx]     = colors_BGR[palette_idx];       // Blue
    img[idx + 1] = colors_BGR[palette_idx + 1];   // Green
    img[idx + 2] = colors_BGR[palette_idx + 2];   // Red
}
)CLC";

const char* changePaletteRGBAOpenCLKernelSource = R"CLC(
__kernel void changePaletteRGBA_kernel(
    __global uchar* img,
    const int nPixels,
    __global const uchar* colors_BGR,
    const int num_colors
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    uchar r = img[idx];
    uchar g = img[idx + 1];
    uchar b = img[idx + 2];

    int min_distance = INT_MAX;
    int nearest_color_idx = 0;

    for (int i = 0; i < num_colors; ++i) {
        int palette_idx = i * 3;

        uchar pb = colors_BGR[palette_idx];
        uchar pg = colors_BGR[palette_idx + 1];
        uchar pr = colors_BGR[palette_idx + 2];

        int db = (int)b - (int)pb;
        int dg = (int)g - (int)pg;
        int dr = (int)r - (int)pr;

        int distance = db * db + dg * dg + dr * dr;

        if (distance < min_distance) {
            min_distance = distance;
            nearest_color_idx = i;
        }
    }

    int palette_idx = nearest_color_idx * 3;
    img[idx]     = colors_BGR[palette_idx + 2];   // Red
    img[idx + 1] = colors_BGR[palette_idx + 1];   // Green
    img[idx + 2] = colors_BGR[palette_idx];       // Blue
}
)CLC";
