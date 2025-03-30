#include "imageBlackAndWhite.h"
#include "image.h"
#include "blackAndWhite_launcher.cuh"


IBlackAndWhiteWorker::IBlackAndWhiteWorker(float middle, QObject* parent)
    : ImageEffect(parent), m_middle(middle)
{
}

void IBlackAndWhiteWorker::process() {
    try {
        Image img(m_inputPath);
        
        unsigned char* d_img;
        
        int blockSize = 1024;
        int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaMallocAsync(&d_img, img.getSize(), stream);
        
        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
        
        blackAndWhite(gridSize, blockSize, stream, d_img, img.getSize(), m_middle);
        
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
