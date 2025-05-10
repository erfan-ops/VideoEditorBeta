#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>

class InverseColorsProcessor {
public:
    InverseColorsProcessor(int size);
    ~InverseColorsProcessor();

    // Function pointer to process image with either CUDA or OpenCL
    static void (InverseColorsProcessor::* processFunc)() const;
    static void (InverseColorsProcessor::* processFuncRGBA)() const;

    // Main processing function (no need to pass img or colorsBGR)
    void process() const;
    void processRGBA() const;

    // setters
    void setImage(const unsigned char* img);
    void upload(unsigned char* Dst) const;

    // initializer
    static void init();

private:
    // CUDA
    cudaStream_t m_cudaStream = nullptr;
    unsigned char* d_img = nullptr;

    int gridSize;
    int blockSize;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_imgBuf = nullptr;

    // members
    unsigned char* m_img = nullptr;
    int imgSize;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;
};
