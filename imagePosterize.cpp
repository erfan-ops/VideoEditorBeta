#include "imagePosterize.h"
#include "image.h"
#include "posterize_launcher.cuh"


IPosterizeWorker::IPosterizeWorker(int threshold, QObject* parent)
    : ImageEffect(parent), m_threshold(threshold)
{
}

void IPosterizeWorker::process() {
    try {
        Image img(m_inputPath);

        unsigned char* d_img;

        int blockSize = 1024;
        int gridSize = (img.getSize() + blockSize - 1) / blockSize;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMallocAsync(&d_img, img.getSize(), stream);

        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);

        posterize(gridSize, blockSize, stream, d_img, img.getSize(), m_threshold);

        cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        cudaFreeAsync(d_img, stream);
        cudaStreamDestroy(stream);

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
