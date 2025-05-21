#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class PosterizeProcessor : public BaseProcessor {
public:
    PosterizeProcessor(int size, float thresh);
    ~PosterizeProcessor();

    void process() const;
    void processRGBA() const;

    static void init();

private:
    static bool firstTime;

    int gridSize;
    int blockSize;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;

    // members
    float m_thresh;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;

    static void (PosterizeProcessor::* processFunc)() const;
    static void (PosterizeProcessor::* processFuncRGBA)() const;
};
