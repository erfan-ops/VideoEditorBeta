//#include "imageEditor.cuh"
//#include "effects.cuh"
//
//#include <cuda_runtime.h>
//#include "image.h"
//#include "utils.h"
//
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
