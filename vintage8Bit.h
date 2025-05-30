#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class Vintage8BitProcessor : public BaseProcessor {
public:
    Vintage8BitProcessor(int size, int width, int height, int pixelWidth, int pixelHeight, float thresh);
    ~Vintage8BitProcessor();

    void process() const;
    void processRGBA() const;

    static void init();

private:
    static bool firstTime;

    int gridSize;
    int blockSize;
    dim3 gridDim;
    dim3 blockDim;
    int roundGridSize;
    unsigned char* d_colors_BGR;

    // OpenCL
    static cl_kernel m_changePaletteOpenclKernel;
    static cl_kernel m_changePaletteOpenclKernelRGBA;
    static cl_kernel m_censorOpenclKernel;
    static cl_kernel m_censorOpenclKernelRGBA;
    static cl_kernel m_posterizeOpenclKernel;
    static cl_kernel m_posterizeOpenclKernelRGBA;
    cl_mem m_colorsBuf;

    // members
    int m_nPixels;
    int m_width;
    int m_height;
    int m_pixelWidth;
    int m_pixelHeight;
    float m_thresh;
    int m_nColors;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;

    static void (Vintage8BitProcessor::* processFunc)() const;
    static void (Vintage8BitProcessor::* processFuncRGBA)() const;

    static unsigned char colors_BGR[42];
};
