#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class MagicEyeProcessor : public BaseProcessor {
public:
    MagicEyeProcessor(int size, int nPixels, float middle);
    ~MagicEyeProcessor();

    void process() const;
    void processRGBA() const { std::cerr << "not implemented!\n"; }

    static void init();

private:
    static bool firstTime;

    int gridSize;
    int blockSize;
    unsigned char* d_noise;

    // OpenCL
    size_t globalSize;
    static cl_kernel m_subtractKernel;
    static cl_kernel m_noiseKernel;
    static cl_kernel m_bwKernel;
    cl_mem m_noiseBuf;

    // members
    int m_nPixels;
    float m_middle;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const { std::cerr << "not implemented!\n"; }
    void processRGBA_OpenCL() const { std::cerr << "not implemented!\n"; }

    static void (MagicEyeProcessor::* processFunc)() const;
};
