#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class SoftPaletteProcessor : public BaseProcessor {
public:
    SoftPaletteProcessor(int nPixels, int size, unsigned char* colorsBGR, int numColors);
    ~SoftPaletteProcessor();

    void process() const;
    void processRGBA() const;

    static void init();

private:
    unsigned char* d_colorsBGR = nullptr;

    int gridSize = 0;
    int blockSize = 0;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_colorBuf = nullptr;

    int m_nPixels;
    int m_numColors;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;

    static void (SoftPaletteProcessor::* processFunc)() const;
    static void (SoftPaletteProcessor::* processFuncRGBA)() const;
};
