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

const char* hueShiftOpenCLKernelSource = R"CLC(
inline void rgb_to_yiq(float r, float g, float b, __private float* y, __private float* i, __private float* q) {
    *y = 0.299f * r + 0.587f * g + 0.114f * b;
    *i = 0.596f * r - 0.274f * g - 0.322f * b;
    *q = 0.211f * r - 0.523f * g + 0.312f * b;
}

inline void yiq_to_rgb(float y, float i, float q, __private float* r, __private float* g, __private float* b) {
    *r = y + 0.956f * i + 0.621f * q;
    *g = y - 0.272f * i - 0.647f * q;
    *b = y - 1.105f * i + 1.702f * q;
}

__kernel void hueShift_kernel(
    __global uchar* img,
    const int nPixels,
    const float rotationFactor
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    float b = (float)(img[idx])     / 255.0f;
    float g = (float)(img[idx + 1]) / 255.0f;
    float r = (float)(img[idx + 2]) / 255.0f;

    float y, i, q;
    rgb_to_yiq(r, g, b, &y, &i, &q);

    float angle = M_PI_F * rotationFactor;
    float cos_theta = cos(angle);
    float sin_theta = sin(angle);

    float new_i = i * cos_theta - q * sin_theta;
    float new_q = i * sin_theta + q * cos_theta;

    yiq_to_rgb(y, new_i, new_q, &r, &g, &b);

    // Clamp and write back
    r = clamp(r, 0.0f, 1.0f);
    g = clamp(g, 0.0f, 1.0f);
    b = clamp(b, 0.0f, 1.0f);

    img[idx]     = (uchar)(b * 255.0f);
    img[idx + 1] = (uchar)(g * 255.0f);
    img[idx + 2] = (uchar)(r * 255.0f);
}
)CLC";

const char* hueShiftRGBAOpenCLKernelSource = R"CLC(
inline void rgb_to_yiq(float r, float g, float b, __private float* y, __private float* i, __private float* q) {
    *y = 0.299f * r + 0.587f * g + 0.114f * b;
    *i = 0.596f * r - 0.274f * g - 0.322f * b;
    *q = 0.211f * r - 0.523f * g + 0.312f * b;
}

inline void yiq_to_rgb(float y, float i, float q, __private float* r, __private float* g, __private float* b) {
    *r = y + 0.956f * i + 0.621f * q;
    *g = y - 0.272f * i - 0.647f * q;
    *b = y - 1.105f * i + 1.702f * q;
}

__kernel void hueShiftRGBA_kernel(
    __global uchar* img,
    const int nPixels,
    const float rotationFactor
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    float r = (float)(img[idx])     / 255.0f;
    float g = (float)(img[idx + 1]) / 255.0f;
    float b = (float)(img[idx + 2]) / 255.0f;

    float y, i, q;
    rgb_to_yiq(r, g, b, &y, &i, &q);

    float angle = M_PI_F * rotationFactor;
    float cos_theta = cos(angle);
    float sin_theta = sin(angle);

    float new_i = i * cos_theta - q * sin_theta;
    float new_q = i * sin_theta + q * cos_theta;

    yiq_to_rgb(y, new_i, new_q, &r, &g, &b);

    // Clamp and write back
    r = clamp(r, 0.0f, 1.0f);
    g = clamp(g, 0.0f, 1.0f);
    b = clamp(b, 0.0f, 1.0f);

    img[idx]     = (uchar)(r * 255.0f);
    img[idx + 1] = (uchar)(g * 255.0f);
    img[idx + 2] = (uchar)(b * 255.0f);
}
)CLC";

const char* inverseColorsOpenCLKernelSource = R"CLC(
__kernel void inverseColors_kernel(
    __global uchar* img,
    const int size
) {
    int idx = get_global_id(0);
    if (idx >= size) return;

    img[idx] = 255 - img[idx];
}
)CLC";

const char* inverseColorsRGBAOpenCLKernelSource = R"CLC(
__kernel void inverseColorsRGBA_kernel(
    __global uchar* img,
    const int size
) {
    int idx = get_global_id(0);
    if (idx % 4 == 3 || idx >= size) return;

    img[idx] = 255 - img[idx];
}
)CLC";

const char* inverseContrastOpenCLKernelSource = R"CLC(
__kernel void inverseContrast_kernel(__global uchar* img, const int nPixels) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Load RGB components and normalize to [0,1]
    float b = (float)(img[idx])     / 255.0f;
    float g = (float)(img[idx + 1]) / 255.0f;
    float r = (float)(img[idx + 2]) / 255.0f;

    // Compute max and min values
    float max_color = fmax(fmax(r, g), b);
    float min_color = fmin(fmin(r, g), b);

    // Compute original lightness
    float l = 0.5f * (max_color + min_color);

    // Invert lightness
    float inverted_l = 1.0f - l;

    float delta = max_color - min_color;
    if (delta < 1e-6f) {
        // Grayscale: set all channels to inverted lightness
        uchar inv = (uchar)(clamp(inverted_l * 255.0f, 0.0f, 255.0f));
        img[idx]     = inv;
        img[idx + 1] = inv;
        img[idx + 2] = inv;
        return;
    }

    // Compute saturation
    float s = delta / (1.0f - fabs(2.0f * l - 1.0f));

    // Compute new min and max based on inverted lightness
    float tmp = s * (1.0f - fabs(2.0f * inverted_l - 1.0f)) * 0.5f;
    float new_max = inverted_l + tmp;
    float new_min = inverted_l - tmp;

    // Remap RGB values while preserving hue
    float new_r = (r == max_color) ? new_max : new_min + (r - min_color) * (new_max - new_min) / delta;
    float new_g = (g == max_color) ? new_max : new_min + (g - min_color) * (new_max - new_min) / delta;
    float new_b = (b == max_color) ? new_max : new_min + (b - min_color) * (new_max - new_min) / delta;

    img[idx]     = (uchar)(clamp(new_b * 255.0f, 0.0f, 255.0f));
    img[idx + 1] = (uchar)(clamp(new_g * 255.0f, 0.0f, 255.0f));
    img[idx + 2] = (uchar)(clamp(new_r * 255.0f, 0.0f, 255.0f));
}
)CLC";


const char* inverseContrastRGBAOpenCLKernelSource = R"CLC(
__kernel void inverseContrastRGBA_kernel(__global uchar* img, const int nPixels) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    // Load RGB components and normalize to [0,1]
    float r = (float)(img[idx])     / 255.0f;
    float g = (float)(img[idx + 1]) / 255.0f;
    float b = (float)(img[idx + 2]) / 255.0f;

    // Compute max and min values
    float max_color = fmax(fmax(r, g), b);
    float min_color = fmin(fmin(r, g), b);

    // Compute original lightness
    float l = 0.5f * (max_color + min_color);

    // Invert lightness
    float inverted_l = 1.0f - l;

    float delta = max_color - min_color;
    if (delta < 1e-6f) {
        // Grayscale: set all channels to inverted lightness
        uchar inv = (uchar)(clamp(inverted_l * 255.0f, 0.0f, 255.0f));
        img[idx]     = inv;
        img[idx + 1] = inv;
        img[idx + 2] = inv;
        return;
    }

    // Compute saturation
    float s = delta / (1.0f - fabs(2.0f * l - 1.0f));

    // Compute new min and max based on inverted lightness
    float tmp = s * (1.0f - fabs(2.0f * inverted_l - 1.0f)) * 0.5f;
    float new_max = inverted_l + tmp;
    float new_min = inverted_l - tmp;

    // Remap RGB values while preserving hue
    float new_r = (r == max_color) ? new_max : new_min + (r - min_color) * (new_max - new_min) / delta;
    float new_g = (g == max_color) ? new_max : new_min + (g - min_color) * (new_max - new_min) / delta;
    float new_b = (b == max_color) ? new_max : new_min + (b - min_color) * (new_max - new_min) / delta;

    img[idx]     = (uchar)(clamp(new_r * 255.0f, 0.0f, 255.0f));
    img[idx + 1] = (uchar)(clamp(new_g * 255.0f, 0.0f, 255.0f));
    img[idx + 2] = (uchar)(clamp(new_b * 255.0f, 0.0f, 255.0f));
}
)CLC";

const char* lensFilterOpenCLKernelSource = R"CLC(
__kernel void lensFilter_kernel(
    __global uchar* img,
    const int size,
    __global const float* passThreshValues
) {
    int idx = get_global_id(0);
    if (idx >= size) return;

    img[idx] = passThreshValues[idx % 3] * img[idx];
}
)CLC";

const char* lensFilterRGBAOpenCLKernelSource = R"CLC(
__kernel void lensFilterRGBA_kernel(
    __global uchar* img,
    const int size,
    __global const float* passThreshValues
) {
    int idx = get_global_id(0);
    if (idx >= size) return;

    img[idx] = passThreshValues[idx % 4] * img[idx];
}
)CLC";

const char* monoChromeOpenCLKernelSource = R"CLC(
__kernel void monoChrome_kernel(
    __global uchar* img,
    const int nPixels
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    uchar m = (uchar)(0.114f * (float)(img[idx]) +
                      0.587f * (float)(img[idx + 1]) +
                      0.299f * (float)(img[idx + 2]));

    img[idx] = m;
    img[idx + 1] = m;
    img[idx + 2] = m;
}
)CLC";

const char* monoChromeRGBAOpenCLKernelSource = R"CLC(
__kernel void monoChromeRGBA_kernel(
    __global uchar* img,
    const int nPixels
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    uchar m = (uchar)(0.114f * (float)(img[idx + 2]) +
                      0.587f * (float)(img[idx + 1]) +
                      0.299f * (float)(img[idx]));

    img[idx] = m;
    img[idx + 1] = m;
    img[idx + 2] = m;
}
)CLC";

const char* monoMaskOpenCLKernelSource = R"CLC(
__kernel void monoMask_kernel(
    __global uchar* img,
    const int nPixels,
    __global const uchar* colors_BGR,
    const int num_colors
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3; // RGB index

    // Compute grayscale mediant value
    float mediant = (0.114f * (float)img[idx] +
                     0.587f * (float)img[idx + 1] +
                     0.299f * (float)img[idx + 2]) / 255.0f;

    // Determine gradient segment
    float segment_size = 1.0f / (float)(num_colors - 1);
    int segment_index = (int)(mediant / segment_size);
    if (segment_index > num_colors - 2) {
        segment_index = num_colors - 2;
    }

    // Compute blending factor
    float segment_start = (float)segment_index * segment_size;
    float segment_end = (float)(segment_index + 1) * segment_size;
    float scale_factor = (mediant - segment_start) / (segment_end - segment_start);

    // Blend colors for B, G, R
    for (int i = 0; i < 3; ++i) {
        uchar color_start = colors_BGR[segment_index * 3 + i];
        uchar color_end = colors_BGR[(segment_index + 1) * 3 + i];
        img[idx + i] = (uchar)((float)color_start +
                              ((float)(color_end - color_start) * scale_factor));
    }
}
)CLC";

const char* monoMaskRGBAOpenCLKernelSource = R"CLC(
__kernel void monoMaskRGBA_kernel(
    __global uchar* img,                // RGBA image
    const int nPixels,                  // Number of pixels
    __global const uchar* colors_BGR,   // Color palette (in BGR)
    const int num_colors                // Number of palette entries
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4; // RGBA image: 4 channels per pixel

    // Compute grayscale mediant from RGB
    float mediant = (0.299f * (float)(img[idx]) +        // R
                     0.587f * (float)(img[idx + 1]) +    // G
                     0.114f * (float)(img[idx + 2]))     // B
                     / 255.0f;

    float segment_size = 1.0f / (float)(num_colors - 1);
    int segment_index = (int)(mediant / segment_size);
    segment_index = min(segment_index, num_colors - 2);

    float segment_start = segment_index * segment_size;
    float segment_end = (segment_index + 1) * segment_size;
    float scale_factor = (mediant - segment_start) / (segment_end - segment_start);

    // Blend BGR palette → into RGB output (reversed index)
    for (int i = 0; i < 3; ++i) {
        uchar color_start = colors_BGR[segment_index * 3 + (2 - i)];
        uchar color_end = colors_BGR[(segment_index + 1) * 3 + (2 - i)];
        img[idx + i] = (uchar)((float)color_start + ((float)(color_end - color_start)) * scale_factor);
    }
}
)CLC";

const char* posterizeOpenCLKernelSource = R"CLC(
__kernel void posterize_kernel(
    __global uchar* img,
    const int size,
    const float thresh
) {
    int idx = get_global_id(0);
    if (idx >= size) return;

    float halfThresh = thresh / 2.0f;
    float colorValue = (float)(img[idx]) + halfThresh;
    float result_value = colorValue - fmod(colorValue, thresh);

    img[idx] = (uchar)(fmin(result_value, 255.0f));
}
)CLC";

const char* posterizeRGBAOpenCLKernelSource = R"CLC(
__kernel void posterizeRGBA_kernel(
    __global uchar* img,
    const int size,
    const float thresh
) {
    int idx = get_global_id(0);
    if (idx >= size) return;
    if (idx % 4 == 3) return; // skip alpha

    float halfThresh = thresh / 2.0f;
    float colorValue = (float)(img[idx]) + halfThresh;
    float result_value = colorValue - fmod(colorValue, thresh);
    
    img[idx] = (uchar)(fmin(result_value, 255.0f));
}
)CLC";

const char* pixelateOpenCLKernelSource = R"CLC(
__kernel void pixelate_kernel(
    __global uchar* img,
    const int rows, const int cols,
    const int pixelWidth, const int pixelHeight,
    const int xBound, const int yBound
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= xBound || y >= yBound) {
        return;
    }

    int blockY = y * pixelHeight;
    int blockX = x * pixelWidth;

    // Calculate the block's end position
    int blockEndX = min(blockX + pixelWidth, cols);
    int blockEndY = min(blockY + pixelHeight, rows);

    // Accumulate the sum of colors in the block
    ulong sum_colors[3] = { 0, 0, 0 }; // Use ulong to match CUDA size_t
    int pixelCount = 0;

    for (int yy = blockY; yy < blockEndY; ++yy) {
        int yc = yy * cols;
        for (int xx = blockX; xx < blockEndX; ++xx) {
            int idx = (yc + xx) * 3;
            for (int c = 0; c < 3; ++c) {
                sum_colors[c] += img[idx + c];
            }
            pixelCount++;
        }
    }

    // Calculate the average color
    uchar avg_colors[3] = { 0, 0, 0 };
    for (int c = 0; c < 3; ++c) {
        avg_colors[c] = (uchar)(sum_colors[c] / pixelCount);
    }

    // Apply the average color to the entire block
    for (int yy = blockY; yy < blockEndY; ++yy) {
        int yc = yy * cols;
        for (int xx = blockX; xx < blockEndX; ++xx) {
            int idx = (yc + xx) * 3;
            for (int c = 0; c < 3; ++c) {
                img[idx + c] = avg_colors[c];
            }
        }
    }
}
)CLC";

const char* pixelateRGBAOpenCLKernelSource = R"CLC(
__kernel void pixelateRGBA_kernel(
    __global uchar* img,
    const int rows, const int cols,
    const int pixelWidth, const int pixelHeight,
    const int xBound, const int yBound
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= xBound || y >= yBound) {
        return;
    }

    int blockY = y * pixelHeight;
    int blockX = x * pixelWidth;

    // Calculate the block's end position
    int blockEndX = min(blockX + pixelWidth, cols);
    int blockEndY = min(blockY + pixelHeight, rows);

    // Accumulate the sum of colors in the block
    ulong sum_colors[4] = { 0, 0, 0, 0 }; // Use ulong to match CUDA size_t
    int pixelCount = 0;

    for (int yy = blockY; yy < blockEndY; ++yy) {
        int yc = yy * cols;
        for (int xx = blockX; xx < blockEndX; ++xx) {
            int idx = (yc + xx) * 4;
            for (int c = 0; c < 4; ++c) {
                sum_colors[c] += img[idx + c];
            }
            pixelCount++;
        }
    }

    // Calculate the average color
    uchar avg_colors[4] = { 0, 0, 0, 0 };
    for (int c = 0; c < 4; ++c) {
        avg_colors[c] = (uchar)(sum_colors[c] / pixelCount);
    }

    // Apply the average color to the entire block
    for (int yy = blockY; yy < blockEndY; ++yy) {
        int yc = yy * cols;
        for (int xx = blockX; xx < blockEndX; ++xx) {
            int idx = (yc + xx) * 4;
            for (int c = 0; c < 4; ++c) {
                img[idx + c] = avg_colors[c];
            }
        }
    }
}
)CLC";

const char* outlinesOpenCLKernelSource = R"CLC(
__kernel void outlines_kernel(
    __global uchar* img,
    __global const uchar* img_copy,
    const int rows, const int cols,
    const int shiftX, const int shiftY
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    int idx = (y * cols + x) * 3;

    if (x < cols - shiftX && y < rows - shiftY) {
        int shiftedIdx = idx + 3 * (shiftY * cols + shiftX);

        for (int c = 0; c < 3; c++) {
            int color_idx = idx + c;
            img[color_idx] = (uchar)(abs((short)(img_copy[color_idx]) - img_copy[shiftedIdx + c]));
        }
    }
    else {
        img[idx++] = 0;
        img[idx++] = 0;
        img[idx] = 0;
    }
}
)CLC";

const char* outlinesRGBAOpenCLKernelSource = R"CLC(
__kernel void outlinesRGBA_kernel(
    __global uchar* img,
    __global const uchar* img_copy,
    const int rows, const int cols,
    const int shiftX, const int shiftY
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    int idx = (y * cols + x) * 4;

    if (x < cols - shiftX && y < rows - shiftY) {
        int shiftedIdx = idx + 4 * (shiftY * cols + shiftX);

        for (int c = 0; c < 3; c++) {
            int color_idx = idx + c;
            img[color_idx] = (uchar)(abs((short)(img_copy[color_idx]) - img_copy[shiftedIdx + c]));
        }
    }
    else {
        img[idx++] = 0;
        img[idx++] = 0;
        img[idx] = 0;
    }
}
)CLC";

const char* subtractOpenCLKernelSource = R"CLC(
__kernel void subtract_kernel(
    __global uchar* img1,
    __global const uchar* img2,
    const int nPixels
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    for (int c = 0; c < 3; c++) {
        int color_idx = idx + c;
        short diff = (short)(img1[color_idx]) - (short)(img2[color_idx]);
        img1[color_idx] = (uchar)(abs(diff));
    }
}
)CLC";

const char* subtractRGBAOpenCLKernelSource = R"CLC(
__kernel void subtractRGBA_kernel(
    __global uchar* img1,
    __global const uchar* img2,
    const int nPixels
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    for (int c = 0; c < 3; c++) {
        int color_idx = idx + c;
        short diff = (short)(img1[color_idx]) - (short)(img2[color_idx]);
        img1[color_idx] = (uchar)(abs(diff));
    }
}
)CLC";

const char* binaryNoiseOpenCLKernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void binaryNoise_kernel(
    __global uchar* img,
    const int nPixels,
    const uint seed
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Use the built-in random function (if available)
    double r = fract(sin((double)(seed + pIdx)) * 43758.5453, &r);
    uchar c = (r > 0.5) ? 255 : 0;

    img[idx]     = c;
    img[idx + 1] = c;
    img[idx + 2] = c;
}
)CLC";

const char* radialBlurOpenCLKernelSource = R"CLC(
__kernel void radialBlur_kernel(
    __global uchar* img,
    __global const uchar* img_copy,
    const int rows,
    const int cols,
    const float centerX,
    const float centerY,
    const int blurRadius,
    const float intensity
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    float dirX = (float)x - centerX;
    float dirY = (float)y - centerY;

    float length = dirX * dirX + dirY * dirY;
    if (length > 0.0f) {
        length = sqrt(length);
        dirX /= length;
        dirY /= length;
    }

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    int count = 0;

    for (int i = -blurRadius; i <= blurRadius; ++i) {
        int sampleX = (int)(x + dirX * (float)i * intensity);
        int sampleY = (int)(y + dirY * (float)i * intensity);

        sampleX = clamp(sampleX, 0, cols - 1);
        sampleY = clamp(sampleY, 0, rows - 1);

        int idx = (sampleY * cols + sampleX) * 3;
        sumR += img_copy[idx];
        sumG += img_copy[idx + 1];
        sumB += img_copy[idx + 2];
        count++;
    }

    uchar avgR = (uchar)(sumR / (float)count);
    uchar avgG = (uchar)(sumG / (float)count);
    uchar avgB = (uchar)(sumB / (float)count);

    int idx = (y * cols + x) * 3;
    img[idx]     = avgR;
    img[idx + 1] = avgG;
    img[idx + 2] = avgB;
}
)CLC";

const char* radialBlurRGBAOpenCLKernelSource = R"CLC(
__kernel void radialBlurRGBA_kernel(
    __global uchar* img,
    __global const uchar* img_copy,
    const int rows,
    const int cols,
    const float centerX,
    const float centerY,
    const int blurRadius,
    const float intensity
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= cols || y >= rows) return;

    float dirX = (float)x - centerX;
    float dirY = (float)y - centerY;

    float length = dirX * dirX + dirY * dirY;
    if (length > 0.0f) {
        length = sqrt(length);
        dirX /= length;
        dirY /= length;
    }

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    int count = 0;

    for (int i = -blurRadius; i <= blurRadius; ++i) {
        int sampleX = (int)(x + dirX * (float)i * intensity);
        int sampleY = (int)(y + dirY * (float)i * intensity);

        sampleX = clamp(sampleX, 0, cols - 1);
        sampleY = clamp(sampleY, 0, rows - 1);

        int idx = (sampleY * cols + sampleX) * 4;
        sumR += img_copy[idx];
        sumG += img_copy[idx + 1];
        sumB += img_copy[idx + 2];
        count++;
    }

    uchar avgR = (uchar)(sumR / (float)count);
    uchar avgG = (uchar)(sumG / (float)count);
    uchar avgB = (uchar)(sumB / (float)count);

    int idx = (y * cols + x) * 4;
    img[idx]     = avgR;
    img[idx + 1] = avgG;
    img[idx + 2] = avgB;
}
)CLC";
