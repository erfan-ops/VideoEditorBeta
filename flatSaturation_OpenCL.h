#pragma once

const char* flatSaturationOpenCLKernelSource = R"CLC(
inline float hueToRGB(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

__kernel void flatSaturation_kernel(
    __global uchar* img,
    const int nPixels,
    const float saturation
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 3;

    // Read and normalize RGB (assuming BGR layout like OpenCV)
    float r = img[idx + 2] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx] / 255.0f;

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

    // Set fixed saturation
    s = saturation;

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
    img[idx] = (uchar)(clamp(b_out * 255.0f, 0.0f, 255.0f));
}
)CLC";

const char* flatSaturationRGBAOpenCLKernelSource = R"CLC(
inline float hueToRGB(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

__kernel void flatSaturationRGBA_kernel(
    __global uchar* img,
    const int nPixels,
    const float saturation
) {
    int pIdx = get_global_id(0);
    if (pIdx >= nPixels) return;

    int idx = pIdx * 4;

    // Read and normalize RGB (assuming BGR layout like OpenCV)
    float r = img[idx] / 255.0f;
    float g = img[idx + 1] / 255.0f;
    float b = img[idx + 2] / 255.0f;

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

    // Set fixed saturation
    s = saturation;

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
    img[idx] = (uchar)(clamp(r_out * 255.0f, 0.0f, 255.0f));
    img[idx + 1] = (uchar)(clamp(g_out * 255.0f, 0.0f, 255.0f));
    img[idx + 2] = (uchar)(clamp(b_out * 255.0f, 0.0f, 255.0f));
}
)CLC";