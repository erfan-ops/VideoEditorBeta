#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>

class BlackAndWhiteProcessor {
public:
    BlackAndWhiteProcessor(int nPixels, int size, float middle);
    ~BlackAndWhiteProcessor();

    // Function pointer to process image with either CUDA or OpenCL
    static void (BlackAndWhiteProcessor::* processFunc)() const;
    static void (BlackAndWhiteProcessor::* processFuncRGBA)() const;

    // Main processing function (no need to pass img or colorsBGR)
    void process() const;
    void processRGBA() const;

    // setters
    void setImage(unsigned char* img, int size);

    // getters
    unsigned char* getImage();

    // initializer
    static void init();

private:
    // CUDA
    cudaStream_t m_cudaStream = nullptr;
    unsigned char* d_img = nullptr;

    int gridSize = 0;
    int blockSize = 0;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_imgBuf = nullptr;

    // members
    unsigned char* m_img = nullptr;
    float m_middle{ 0 };

    int m_nPixels{ 0 };
    int imgSize{ 0 };

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;
};
