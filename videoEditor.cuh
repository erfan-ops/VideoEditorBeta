#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


__host__ void videoVintage8bit(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short& pixelWidth,
    const unsigned short& pixelHeight,
    const unsigned char* colors_BGR,
    const size_t& nColors,
    const unsigned char& threshold,
    const unsigned short& lineWidth,
    const unsigned char& lineDarkeningThresh
);
