#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>
#include "baseProcessor.h"

class OutlinesProcessor : public BaseProcessor {
public:
    OutlinesProcessor(int size, int width, int height, int xShift, int yShift);
    ~OutlinesProcessor();

    void upload(const unsigned char* Src) { (this->*uploadFunc)(Src); };
    void download(unsigned char* Dst) const { (this->*downloadFunc)(Dst); };

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
    int m_pixelWidth;
    int m_pixelHeight;
    int m_width;
    int m_height;
    int m_xShift;
    int m_yShift;

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

    static void (OutlinesProcessor::* uploadFunc)(const unsigned char* Src);
    static void (OutlinesProcessor::* downloadFunc)(unsigned char* Dst) const;

    static void (OutlinesProcessor::* processFunc)() const;
    static void (OutlinesProcessor::* processFuncRGBA)() const;
};
