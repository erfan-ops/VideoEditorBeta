#include "imageVintage8bit.h"
#include "image.h"
#include "vintage8bit_launcher.cuh"


void IVintage8bitWorker::process() {
    try {
        Image img(m_inputPath);

        unsigned char* d_img;
        unsigned char* d_colorsBGR;

        unsigned char colorsBGR[] = {
            64, 9, 67,
            61, 70, 133,
            59, 131, 197,
            58, 124, 127,
            61, 64, 61,
            122, 188, 191,
            122, 194, 255,
            121, 246, 255,
            187, 251, 254,
            125, 134, 197,
        };



        dim3 blockDim(32, 32);
        dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);

        int blockSize = 1024;
        int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
        int roundGridSize = (img.getSize() + blockSize - 1) / blockSize;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMallocAsync(&d_img, img.getSize(), stream);
        cudaMallocAsync(&d_colorsBGR, 30 * sizeof(char), stream);

        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_colorsBGR, colorsBGR, 30 * sizeof(char), cudaMemcpyHostToDevice, stream);

        vintage8bit(
            gridDim, blockDim, gridSize, blockSize, roundGridSize, stream, d_img,
            m_pixelWidth, m_pixelHeight, m_thresh, d_colorsBGR, 10,
            img.getWidth(), img.getHeight(), img.getNumPixels(), img.getSize()
        );

        cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        cudaFreeAsync(d_img, stream);
        cudaFreeAsync(d_colorsBGR, stream);
        cudaStreamDestroy(stream);

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
