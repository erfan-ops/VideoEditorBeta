#pragma once

#include <CL/cl.h>
#include <cuda_runtime.h>

class BlurProcessor {
public:
    BlurProcessor(int size, int width, int height, int radius);
    ~BlurProcessor();

    // Function pointer to process image with either CUDA or OpenCL
    static void (BlurProcessor::* processFunc)() const;
    static void (BlurProcessor::* processFuncRGBA)() const;

    // Main processing function (no need to pass img or colorsBGR)
    void process() const;
    void processRGBA() const;

    // setters
    void setImage(unsigned char* img);

    // getters
    unsigned char* getImage();

    // initializer
    static void init();

private:
    // CUDA
    cudaStream_t m_cudaStream = nullptr;
    unsigned char* d_img = nullptr;
    unsigned char* d_img_copy = nullptr;

    dim3 gridDim;
    dim3 blockDim;

    // OpenCL
    static cl_kernel m_openclKernel;
    static cl_kernel m_openclKernelRGBA;
    cl_mem m_imgBuf = nullptr;
    cl_mem m_imgCopyBuf = nullptr;

    // members
    unsigned char* m_img = nullptr;
    int m_radius{ 0 };
    int imgSize{ 0 };
    int m_width;
    int m_height;

    // Allocate buffers
    void allocateCUDA();
    void allocateOpenCL();

    // processors
    void processCUDA() const;
    void processOpenCL() const;

    void processRGBA_CUDA() const;
    void processRGBA_OpenCL() const;
};
