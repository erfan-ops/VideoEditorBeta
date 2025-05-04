#include <CL/cl.h>
#include <string>

const char* blendKernelSource = R"CLC(
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


