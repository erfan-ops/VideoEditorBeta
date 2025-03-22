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

__host__ void imageChangePalette(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char* colors_BGR,
    const int num_colors
);

__host__ void imageRadialBlur(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const int blurRadius,
    const float intensity,
    float centerX = -1,
    float centerY = -1
);

__host__ void imageReverseContrast(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void imageHueShift(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const float hueShift
);

__host__ void imageInverseColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void imageMonoChrome(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void imageOutLines(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    int shiftX = 1, int shiftY = 1
);

__host__ void imageTrueOutLines(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const int thresh = 1
);

__host__ void imageLensFilter(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const float* __restrict passThreshValues
);

__host__ void imageBlackNWhite(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const float middle = 127.5f
);

__host__ void imageBlur(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const int blurRadius,
    const int blending = 0
);
