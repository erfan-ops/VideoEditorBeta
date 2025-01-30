#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__host__ void videoVintage8bit(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    int pixelWidth,
    int pixelHeight,
    const unsigned char* color_BGR,
    int threshold,
    int lineWidth,
    int lineDarkeningThresh
);

__host__ void videoVintage8bit2(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short& pixelWidth,
    const unsigned short& pixelHeight,
    const unsigned char* color1,
    const unsigned char* color2,
    const unsigned char* color3,
    const unsigned char& threshold,
    const unsigned short& lineWidth,
    const unsigned char& lineDarkeningThresh
);



__host__ void videoVintage8bit3(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short& pixelWidth,
    const unsigned short& pixelHeight,
    const unsigned char* color1,
    const unsigned char* color2,
    const unsigned char* color3,
    const unsigned char& threshold,
    const unsigned short& lineWidth,
    const unsigned char& lineDarkeningThresh
);
