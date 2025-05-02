#include "imageChangePalette.h"
#include "image.h"
#include "changePalette_launcher.cuh"


void IChangePaletteWorker::process() {
    try {
        Image img(m_inputPath);

        unsigned char* d_img;
        unsigned char* d_colorsBGR;

        int blockSize = 1024;
        int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMallocAsync(&d_img, img.getSize(), stream);
        cudaMallocAsync(&d_colorsBGR, 3ULL * sizeof(unsigned char) * m_numColors, stream);

        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_colorsBGR, m_colorsBGR, 3ULL * sizeof(unsigned char) * m_numColors, cudaMemcpyHostToDevice, stream);

        changePalette(gridSize, blockSize, stream, d_img, img.getNumPixels(), d_colorsBGR, m_numColors);

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
