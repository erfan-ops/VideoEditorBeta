#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class FlatSaturationProcessor : public BaseProcessor {
public:
    FlatSaturationProcessor(int size, int nPixels, float thresh);
    ~FlatSaturationProcessor();

    void process() const;
    void processRGBA() const;

    static void init();

private:
    static bool firstTime;

    int gridSize;
    int blockSize;

    // OpenCL
    size_t globalWorkSize;
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;

    // members
    float m_thresh;
    int m_nPixels;

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;

    static void (FlatSaturationProcessor::* processFunc)() const;
    static void (FlatSaturationProcessor::* processFuncRGBA)() const;
};
