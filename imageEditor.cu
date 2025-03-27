//#include "imageEditor.cuh"
//#include "effects.cuh"
//
//#include <cuda_runtime.h>
//#include "image.h"
//#include "utils.h"
//
//
//
//__host__ void imagePixelate(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const unsigned short pixelWidth,
//    const unsigned short pixelHeight
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    dim3 blockDim(32, 32);
//    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);
//
//    cudaMalloc(&d_img, img.getSize());
//    cudaMemcpy(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice);
//
//    pixelate_kernel<<<gridDim, blockDim>>>(d_img, img.getHeight(), img.getWidth(), pixelWidth, pixelHeight);
//
//    cudaMemcpy(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost);
//    cudaFree(d_img);
//
//    img.save(outputPath);
//}
//
//__host__ void imageCensor(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const unsigned short pixelWidth,
//    const unsigned short pixelHeight
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    dim3 blockDim(32, 32);
//    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);
//
//    cudaMalloc(&d_img, img.getSize());
//    cudaMemcpy(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice);
//
//    censor_kernel<<<gridDim, blockDim>>>(d_img, img.getHeight(), img.getWidth(), pixelWidth, pixelHeight);
//
//    cudaMemcpy(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost);
//    cudaFree(d_img);
//
//    img.save(outputPath);
//}
//
//__host__ void imageRoundColors(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const unsigned char thresh
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaMalloc(&d_img, img.getSize());
//    cudaMemcpy(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice);
//
//    roundColors_kernel<<<gridSize, blockSize>>>(d_img, img.getSize(), thresh);
//
//    cudaMemcpy(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost);
//    cudaFree(d_img);
//
//    img.save(outputPath);
//}
//
//__host__ void imageMonoMask(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const unsigned char* colors_BGR,
//    const int num_colors
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//    unsigned char* d_colors;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//    static const size_t colorsSize = 3ull * num_colors * sizeof(unsigned char);
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//    cudaMallocAsync(&d_colors, colorsSize, stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//    cudaMemcpyAsync(d_colors, colors_BGR, colorsSize, cudaMemcpyHostToDevice, stream);
//
//    dynamicColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize(), d_colors, num_colors);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaFreeAsync(d_colors, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageChangePalette(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const unsigned char* colors_BGR,
//    const int num_colors
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//    unsigned char* d_colors;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//    static const size_t colorsSize = 3ull * num_colors * sizeof(unsigned char);
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//    cudaMallocAsync(&d_colors, colorsSize, stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//    cudaMemcpyAsync(d_colors, colors_BGR, colorsSize, cudaMemcpyHostToDevice, stream);
//
//    nearestColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize(), d_colors, num_colors);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaFreeAsync(d_colors, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageRadialBlur(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const int blurRadius,
//    const float intensity,
//    float centerX,
//    float centerY
//) {
//    Image img(inputPath);
//
//    if (centerX == -1) centerX = img.getWidth() * 0.5f;
//    if (centerY == -1) centerY = img.getHeight() * 0.5f;
//
//    unsigned char* d_img;
//
//    dim3 blockDim(32, 32);
//    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//
//    radial_blur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, img.getHeight(), img.getWidth(), centerX, centerY, blurRadius, intensity);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageReverseContrast(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//
//    reverse_contrast<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize());
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageHueShift(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const float hueShift
//) {
//    static const float rotationFactor = 2.0f * hueShift;
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//
//    shift_hue_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize(), rotationFactor);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageInverseColors(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//
//    inverseColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize());
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageMonoChrome(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//
//    monoChrome_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize());
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageOutLines(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    int shiftX, int shiftY
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//    unsigned char* d_img_copy;
//
//    dim3 blockDim(32, 32);
//    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//    cudaMallocAsync(&d_img_copy, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//    cudaMemcpyAsync(d_img_copy, d_img, img.getSize(), cudaMemcpyDeviceToDevice, stream);
//
//    outlines_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, img.getHeight(), img.getWidth(), shiftX, shiftY);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaFreeAsync(d_img_copy, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageTrueOutLines(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const int thresh
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//    unsigned char* d_img_copy;
//
//    dim3 blockDim(32, 32);
//    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//    cudaMallocAsync(&d_img_copy, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//    cudaMemcpyAsync(d_img_copy, d_img, img.getSize(), cudaMemcpyDeviceToDevice, stream);
//
//    fastBlur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, img.getHeight(), img.getWidth(), thresh);
//    subtract_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, d_img_copy, img.getNumPixels());
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaFreeAsync(d_img_copy, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageLensFilter(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const float* __restrict passThreshValues
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//    float* d_passThreshValues;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    static constexpr size_t color_size = 3ULL * sizeof(float);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//    cudaMallocAsync(&d_passThreshValues, color_size, stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//    cudaMemcpyAsync(d_passThreshValues, passThreshValues, color_size, cudaMemcpyHostToDevice, stream);
//
//    passColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize(), d_passThreshValues);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaFreeAsync(d_passThreshValues, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageBlackNWhite(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const float middle
//) {
//    Image img(inputPath);
//
//    unsigned char* d_img;
//
//    int blockSize = 1024;
//    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//
//    blackNwhite_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize(), middle);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
//
//__host__ void imageBlur(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const int blurRadius,
//    const int blending
//) {
//    using KernelFunction = void (*)(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius);
//    KernelFunction blur_func = nullptr;
//    if (blending == 0) {
//        blur_func = &fastBlur_kernel;
//    }
//    else if (blending == 1) {
//        blur_func = &trueBlur_kernel;
//    }
//    else {
//        blur_func = &fastBlur_kernel;
//    }
//
//    Image img(inputPath);
//
//    unsigned char* d_img;
//    unsigned char* d_img_copy;
//
//    dim3 blockDim(32, 32);
//    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);
//
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//
//    cudaMallocAsync(&d_img, img.getSize(), stream);
//    cudaMallocAsync(&d_img_copy, img.getSize(), stream);
//
//    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
//    cudaMemcpyAsync(d_img_copy, d_img, img.getSize(), cudaMemcpyDeviceToDevice, stream);
//
//    blur_func<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, img.getHeight(), img.getWidth(), blurRadius);
//
//    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
//
//    cudaStreamSynchronize(stream);
//
//    cudaFreeAsync(d_img, stream);
//    cudaFreeAsync(d_img_copy, stream);
//    cudaStreamDestroy(stream);
//
//    img.save(outputPath);
//}
