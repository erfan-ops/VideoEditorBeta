#include "imageLensFilter.h"
#include "image.h"
#include "lensFilter_launcher.cuh"


void ILensFilterWorker::process() {
    try {
        Image img(m_inputPath);

        unsigned char* d_img;
        float* d_passThreshValues;

        int blockSize = 1024;
        int gridSize = (img.getSize() + blockSize - 1) / blockSize;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMallocAsync(&d_img, img.getSize(), stream);
        cudaMallocAsync(&d_passThreshValues, 3 * sizeof(float), stream);

        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_passThreshValues, m_passThreshValues, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        lensFilter(gridSize, blockSize, stream, d_img, img.getSize(), d_passThreshValues);

        cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        cudaFreeAsync(d_img, stream);
        cudaFreeAsync(d_passThreshValues, stream);
        cudaStreamDestroy(stream);

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
