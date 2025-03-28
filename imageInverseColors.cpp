#include "imageInverseColors.h"
#include "image.h"
#include "inverseColors_launcher.cuh"


IInverseColorsWorker::IInverseColorsWorker(QObject* parent)
{
}

void IInverseColorsWorker::process() {
    try {
        Image img(m_inputPath);
        
        unsigned char* d_img;
        
        int blockSize = 1024;
        int gridSize = (img.getSize() + blockSize - 1) / blockSize;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaMallocAsync(&d_img, img.getSize(), stream);
        
        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
        
        inverseColors(gridSize, blockSize, stream, d_img, img.getSize());
        
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
