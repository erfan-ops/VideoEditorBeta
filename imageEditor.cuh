#pragma once

#include <cuda_runtime.h>
#include <string>

__host__ void imagePixelate(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
);

__host__ void imageCensor(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
);

__host__ void imageRoundColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char thresh
);

__host__ void imageMonoMask(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char* colors_BGR,
    const int num_colors
);
