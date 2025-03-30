#include "imageCensor.h"
#include "image.h"
#include "censor_launcher.cuh"


void ICensorWorker::process() {
    try {
        Image img(m_inputPath);

        unsigned char* d_img;

        dim3 blockDim(32, 32);
        dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMallocAsync(&d_img, img.getSize(), stream);
        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);

        censor(gridDim, blockDim, stream, d_img, img.getWidth(), img.getHeight(), m_pixelWidth, m_pixelHeight);

        cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(d_img, stream);
        cudaStreamDestroy(stream);

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
