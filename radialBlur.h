#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class RadialBlurProcessor : public BaseProcessor {
public:
    RadialBlurProcessor(int size, int width, int height, float centerX, float centerY, int blurRadius, float intensity);
    ~RadialBlurProcessor();

    void upload(const unsigned char* Src) { (this->*uploadFunc)(Src); };

    void process() const;
    void processRGBA() const;

    static void init();

private:
    static bool firstTime;

    dim3 gridDim;
    dim3 blockDim;
    unsigned char* d_imgCopy;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_imgCopyBuf;

    // members
    int m_width;
    int m_height;
    float m_centerX;
    float m_centerY;
    int m_blurRadius;
    float m_intensity;

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

    static void (RadialBlurProcessor::* uploadFunc)(const unsigned char* Src);

    static void (RadialBlurProcessor::* processFunc)() const;
    static void (RadialBlurProcessor::* processFuncRGBA)() const;
};
