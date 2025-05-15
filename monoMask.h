#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>

class MonoMaskProcessor {
public:
    MonoMaskProcessor(int nPixels, int size, unsigned char* colorsBGR, int numColors);
    ~MonoMaskProcessor();

    // Function pointer to process image with either CUDA or OpenCL
    static void (MonoMaskProcessor::* processFunc)() const;
    static void (MonoMaskProcessor::* processFuncRGBA)() const;

    // Main processing function (no need to pass img or colorsBGR)
    void process() const;
    void processRGBA() const;

    // setters
    void setImage(const unsigned char* img);

    void upload(unsigned char* Dst) const;

    // initializer
    static void init();

private:
    // CUDA
    cudaStream_t m_cudaStream = nullptr;
    unsigned char* d_img = nullptr;
    unsigned char* d_colorsBGR = nullptr;

    int gridSize = 0;
    int blockSize = 0;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_imgBuf = nullptr;
    cl_mem m_colorBuf = nullptr;

    // Image and color palette members
    unsigned char* m_img = nullptr;

    int m_nPixels{ 0 };
    int m_numColors{ 0 };
    int imgSize{ 0 };

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;
};
