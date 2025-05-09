#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>

class CensorProcessor {
public:
    CensorProcessor(int size, int width, int height, int pixelWidth, int pixelHeight);
    ~CensorProcessor();

    // Function pointer to process image with either CUDA or OpenCL
    static void (CensorProcessor::* processFunc)() const;
    static void (CensorProcessor::* processFuncRGBA)() const;

    // Main processing function (no need to pass img or colorsBGR)
    void process() const;
    void processRGBA() const;

    // setters
    void setImage(const unsigned char* img);
    void upload(unsigned char* Dst);

    // initializer
    static void init();

private:
    // CUDA
    cudaStream_t m_cudaStream = nullptr;
    unsigned char* d_img = nullptr;

    dim3 gridDim;
    dim3 blockDim;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_imgBuf = nullptr;

    // members
    unsigned char* m_img = nullptr;
    int m_pixelWidth;
    int m_pixelHeight;
    int imgSize;
    int m_width;
    int m_height;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;
};
