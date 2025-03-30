#include "imageTrueOutlines.h"
#include "image.h"
#include "trueOutlines_launcher.cuh"


void ITrueOutlinesWorker::process() {
    try {
        Image img(m_inputPath);
        
        unsigned char* d_img;
        unsigned char* d_img_copy;
        
        dim3 blockDim(32, 32);
        dim3 gridDim((img.getWidth() + blockDim.x - 1) / blockDim.x, (img.getHeight() + blockDim.y - 1) / blockDim.y);
        
        int blockSize = 1024;
        int gridSize = (img.getNumPixels() + blockSize - 1) / blockSize;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaMallocAsync(&d_img, img.getSize(), stream);
        cudaMallocAsync(&d_img_copy, img.getSize(), stream);
        
        cudaMemcpyAsync(d_img, img.getData(), img.getSize(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_img_copy, d_img, img.getSize(), cudaMemcpyDeviceToDevice, stream);
        
        trueOutlines(gridSize, blockSize, gridDim, blockDim, stream, d_img, d_img_copy,
                     img.getWidth(), img.getHeight(), img.getNumPixels(), m_thresh);
        
        cudaMemcpyAsync(img.getData(), d_img, img.getSize(), cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        
        cudaFreeAsync(d_img, stream);
        cudaFreeAsync(d_img_copy, stream);
        cudaStreamDestroy(stream);
        
        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
