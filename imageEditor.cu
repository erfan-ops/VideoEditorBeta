#include "imageEditor.cuh"
#include "effects.cuh"

#include <cuda_runtime.h>
#include "image.h"
#include "utils.h"



__host__ void imagePixelate(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
) {
    Image img(inputPath);

    unsigned char* d_img;

    dim3 blockDim(32, 32);
    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);

    cudaMalloc(&d_img, img.getSize());
    cudaMemcpy(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice);

    pixelate_kernel<<<gridDim, blockDim>>>(d_img, img.getHeight(), img.getWidth(), pixelWidth, pixelHeight);

    cudaMemcpy(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost);
    cudaFree(d_img);

    img.save(outputPath);
}

__host__ void imageCensor(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
) {
    Image img(inputPath);

    unsigned char* d_img;

    dim3 blockDim(32, 32);
    dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);

    cudaMalloc(&d_img, img.getSize());
    cudaMemcpy(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice);

    censor_kernel<<<gridDim, blockDim>>>(d_img, img.getHeight(), img.getWidth(), pixelWidth, pixelHeight);

    cudaMemcpy(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost);
    cudaFree(d_img);

    img.save(outputPath);
}

__host__ void imageRoundColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char thresh
) {
    Image img(inputPath);

    unsigned char* d_img;

    int blockSize = 1024;
    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;

    cudaMalloc(&d_img, img.getSize());
    cudaMemcpy(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice);

    roundColors_kernel<<<gridSize, blockSize>>>(d_img, img.getSize(), thresh);

    cudaMemcpy(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost);
    cudaFree(d_img);

    img.save(outputPath);
}

__host__ void imageMonoMask(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char* colors_BGR,
    const int num_colors
) {
    Image img(inputPath);

    unsigned char* d_img;
    unsigned char* d_colors;

    int blockSize = 1024;
    int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
    static const size_t colorsSize = 3ull * num_colors * sizeof(unsigned char);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocAsync(&d_img, img.getSize(), stream);
    cudaMallocAsync(&d_colors, colorsSize, stream);

    cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_colors, colors_BGR, colorsSize, cudaMemcpyHostToDevice, stream);

    dynamicColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, img.getSize(), d_colors, num_colors);

    cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_img, stream);
    cudaFreeAsync(d_colors, stream);
    cudaStreamDestroy(stream);

    img.save(outputPath);
}
