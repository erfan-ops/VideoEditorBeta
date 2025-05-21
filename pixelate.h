#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class PixelateProcessor : public BaseProcessor {
public:
    PixelateProcessor(int size, int width, int height, int pixelWidth, int pixelHeight);
    ~PixelateProcessor();

    void process() const;
    void processRGBA() const;

    static void init();

private:
    static bool firstTime;

    dim3 gridDim;
    dim3 blockDim;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;

    // members
    int m_pixelWidth;
    int m_pixelHeight;
    int m_width;
    int m_height;
    int m_xBound;
    int m_yBound;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;

    static void (PixelateProcessor::* processFunc)() const;
    static void (PixelateProcessor::* processFuncRGBA)() const;
};
