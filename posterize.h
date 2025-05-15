#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>

class PosterizeProcessor {
public:
    PosterizeProcessor(int size, float thresh);
    ~PosterizeProcessor();

    // Function pointer to process image with either CUDA or OpenCL
    static void (PosterizeProcessor::* processFunc)() const;
    static void (PosterizeProcessor::* processFuncRGBA)() const;

    static void (PosterizeProcessor::* uploadFunc)(const unsigned char* Src);
    static void (PosterizeProcessor::* downloadFunc)(unsigned char* Dst) const;

    // Main processing function (no need to pass img or colorsBGR)
    void process() const;
    void processRGBA() const;

    // setters
    void upload(const unsigned char* Src);
    void download(unsigned char* Dst) const;

    // initializer
    static void init();

private:
    // CUDA
    cudaStream_t m_cudaStream = nullptr;
    unsigned char* d_img = nullptr;

    int gridSize;
    int blockSize;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_imgBuf = nullptr;

    // members
    float m_thresh;
    int imgSize;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    void uploadCUDA(const unsigned char* Src);
    void uploadOpenCL(const unsigned char* Src);

    void downloadCUDA(unsigned char* Dst) const;
    void downloadOpenCL(unsigned char* Dst) const;

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;
};
