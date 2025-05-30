#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class TrueOutlinesProcessor : public BaseProcessor {
public:
    TrueOutlinesProcessor(int size, int width, int height, int radius);
    ~TrueOutlinesProcessor();

    void upload(const unsigned char* Src) { (this->*uploadFunc)(Src); };

    void process() const;
    void processRGBA() const;

    static void init();

private:
    static bool firstTime;

    int gridSize;
    int blockSize;
    dim3 gridDim;
    dim3 blockDim;
    unsigned char* d_imgCopy;

    // OpenCL
    static cl_kernel m_blurOpenclKernel;
    static cl_kernel m_blurOpenclKernelRGBA;
    static cl_kernel m_subtractOpenclKernel;
    static cl_kernel m_subtractOpenclKernelRGBA;
    cl_mem m_imgCopyBuf;

    // members
    int m_radius;
    int m_nPixels;
    int m_width;
    int m_height;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    void uploadCUDA(const unsigned char* Src);
    void uploadOpenCL(const unsigned char* Src);

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;

    static void (TrueOutlinesProcessor::* uploadFunc)(const unsigned char* Src);

    static void (TrueOutlinesProcessor::* processFunc)() const;
    static void (TrueOutlinesProcessor::* processFuncRGBA)() const;
};
