#pragma once

#include <CL/cl.h>

__kernel void blendNearestColors_kernel(
    __global uchar* img,
    const int nPixels,
    __global const uchar* colors_BGR,
    const int num_colors
);
