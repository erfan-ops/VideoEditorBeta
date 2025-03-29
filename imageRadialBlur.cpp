#include "imageRadialBlur.h"
#include "radialBlur_launcher.cuh"
#include "image.h"


IRadialBlurWorker::IRadialBlurWorker(int blurRadius, float intensity, float centerX, float centerY, QObject* parent)
    : ImageEffect(parent), m_blurRadius(blurRadius), m_intensity(intensity), m_centerX(centerX), m_centerY(centerY)
{
}

void IRadialBlurWorker::process() {
    try {
        Image img(m_inputPath);

        if (m_centerX < 0.0f || m_centerX >= img.getWidth()) m_centerX = img.getWidth() * 0.5f;
        if (m_centerY < 0.0f || m_centerY >= img.getHeight()) m_centerY = img.getHeight() * 0.5f;

        unsigned char* d_img;

        dim3 blockDim(32, 32);
        dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMallocAsync(&d_img, img.getSize(), stream);

        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);

        radialBlur(gridDim, blockDim, stream, d_img, img.getWidth(), img.getHeight(), m_centerX, m_centerY, m_blurRadius, m_intensity);

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
