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
