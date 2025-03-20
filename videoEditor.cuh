#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


__host__ void videoVintage8bit(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight,
    const unsigned char* colors_BGR,
    const size_t nColors,
    const unsigned char threshold,
    const unsigned short lineWidth,
    const unsigned char lineDarkeningThresh
);

__host__ void videoRadialBlur(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    int blurRadius,
    float intensity,
    float centerX = -1,
    float centerY = -1
);

__host__ void videoReverseContrast(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void videoShiftHue(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    float hue_shift
);

__host__ void videoOutlines(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    int shiftX = 1, int shiftY = 1
);

__host__ void videoHighlightMotion(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void videoBlur(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const int blurRadius,
    const int blending = 0,
    const float precision = 0.1f
);

__host__ void videoTrueOutlines(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const int thresh = 1
);

__host__ void videoMonoChrome(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void videoMonoMask(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char* colors_BGR,
    const size_t nColors
);

__host__ void videoPassColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const float* __restrict passThresh
);

__host__ void videoPixelate(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
);

__host__ void videoCensor(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
);

__host__ void videoRoundColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char thresh
);

__host__ void videoInverseColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void videoBlackWhite(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);

__host__ void videoMagicEye(
    const std::wstring& inputPath,
    const std::wstring& outputPath
);
