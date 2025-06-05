#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>

#include "globals.h"
#include "utils.h"

class BaseProcessor {
public:
    virtual void process() const = 0;
    virtual void processRGBA() const = 0;

    virtual void upload(const unsigned char* Src) { (this->*uploadFunc)(Src); };
    virtual void download(unsigned char* Dst) const { (this->*downloadFunc)(Dst); };

    static void init();

protected:
    static bool firstTime;

    // CUDA
    cudaStream_t m_cudaStream;
    unsigned char* d_img;

    // OpenCL
    cl_mem m_imgBuf;

    // members
    int imgSize;

    // Allocate buffers
    virtual void allocateCUDA();
    virtual void allocateOpenCL();

    virtual void uploadCUDA(const unsigned char* Src);
    virtual void uploadOpenCL(const unsigned char* Src);

    virtual void downloadCUDA(unsigned char* Dst) const;
    virtual void downloadOpenCL(unsigned char* Dst) const;

    // processors
    virtual void processCUDA() const = 0;
    virtual void processOpenCL() const = 0;

    virtual void processRGBA_CUDA() const = 0;
    virtual void processRGBA_OpenCL() const = 0;

private:
    static void (BaseProcessor::* uploadFunc)(const unsigned char* Src);
    static void (BaseProcessor::* downloadFunc)(unsigned char* Dst) const;
};
