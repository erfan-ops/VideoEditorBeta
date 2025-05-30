#include "trueOutlines.h"
#include "trueOutlines_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

bool TrueOutlinesProcessor::firstTime = true;

void (TrueOutlinesProcessor::* TrueOutlinesProcessor::uploadFunc)(const unsigned char* Src) = nullptr;

void (TrueOutlinesProcessor::* TrueOutlinesProcessor::processFunc)() const = nullptr;
void (TrueOutlinesProcessor::* TrueOutlinesProcessor::processFuncRGBA)() const = nullptr;

cl_kernel TrueOutlinesProcessor::m_blurOpenclKernel = nullptr;
cl_kernel TrueOutlinesProcessor::m_blurOpenclKernelRGBA = nullptr;
cl_kernel TrueOutlinesProcessor::m_subtractOpenclKernel = nullptr;
cl_kernel TrueOutlinesProcessor::m_subtractOpenclKernelRGBA = nullptr;

TrueOutlinesProcessor::TrueOutlinesProcessor(int size, int width, int height, int radius)
    : m_width(width), m_height(height), m_radius(radius), m_nPixels(width * height) {
    imgSize = size;

    if (TrueOutlinesProcessor::firstTime) {
        TrueOutlinesProcessor::init();
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();

        blockSize = 1024;
        gridSize = (m_nPixels + blockSize - 1) / blockSize;

        blockDim = dim3(32, 32);
        gridDim = dim3(
            (this->m_width + blockDim.x - 1) / blockDim.x,
            (this->m_height + blockDim.y - 1) / blockDim.y
        );
    }
    else {
        allocateOpenCL();
    }
}

TrueOutlinesProcessor::~TrueOutlinesProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaFree(d_imgCopy);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
        clReleaseMemObject(m_imgCopyBuf);
    }
}

void TrueOutlinesProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_imgCopy, imgSize, m_cudaStream);
}

void TrueOutlinesProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_imgCopyBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY, imgSize, nullptr, &err);
}

void TrueOutlinesProcessor::uploadCUDA(const unsigned char* Src) {
    cudaMemcpyAsync(d_img, Src, imgSize, cudaMemcpyHostToDevice, m_cudaStream);
    cudaMemcpyAsync(d_imgCopy, d_img, imgSize, cudaMemcpyDeviceToDevice, m_cudaStream);
}

void TrueOutlinesProcessor::uploadOpenCL(const unsigned char* Src) {
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,
        0,
        imgSize,
        Src,
        0, nullptr, nullptr);
    clEnqueueCopyBuffer(globalQueueOpenCL,
        m_imgBuf,
        m_imgCopyBuf,
        0,
        0,
        imgSize,
        0, nullptr, nullptr);
}

void TrueOutlinesProcessor::process() const {
    (this->*processFunc)();
}

void TrueOutlinesProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void TrueOutlinesProcessor::processCUDA() const {
    trueOutlines_CUDA(
        gridSize, blockSize, gridDim, blockDim, m_cudaStream,
        d_img, d_imgCopy, m_width, m_height, m_nPixels, m_radius
    );
}

void TrueOutlinesProcessor::processRGBA_CUDA() const {
    trueOutlinesRGBA_CUDA(
        gridSize, blockSize, gridDim, blockDim, m_cudaStream,
        d_img, d_imgCopy, m_width, m_height, m_nPixels, m_radius
    );
}

void TrueOutlinesProcessor::processOpenCL() const {
    cl_event blurEvent, subtractEvent;

    size_t globalSizeBlur[2] = { (size_t)m_width, (size_t)m_height };
    size_t globalSizeSubtract = m_nPixels;

    clSetKernelArg(m_blurOpenclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_blurOpenclKernel, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_blurOpenclKernel, 2, sizeof(int), &m_height);
    clSetKernelArg(m_blurOpenclKernel, 3, sizeof(int), &m_width);
    clSetKernelArg(m_blurOpenclKernel, 4, sizeof(int), &m_radius);
    
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_blurOpenclKernel,
        2,
        nullptr,
        globalSizeBlur,
        nullptr,
        0, nullptr, &blurEvent);

    clSetKernelArg(m_subtractOpenclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_subtractOpenclKernel, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_subtractOpenclKernel, 2, sizeof(int), &m_nPixels);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_subtractOpenclKernel,
        1,
        nullptr,
        &globalSizeSubtract,
        nullptr,
        0, nullptr, &subtractEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &subtractEvent);

    // Release the kernel event
    clReleaseEvent(blurEvent);
    clReleaseEvent(subtractEvent);
}

void TrueOutlinesProcessor::processRGBA_OpenCL() const {
    cl_event blurEvent, subtractEvent;

    clSetKernelArg(m_blurOpenclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_blurOpenclKernelRGBA, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_blurOpenclKernelRGBA, 2, sizeof(int), &m_height);
    clSetKernelArg(m_blurOpenclKernelRGBA, 3, sizeof(int), &m_width);
    clSetKernelArg(m_blurOpenclKernelRGBA, 4, sizeof(int), &m_radius);

    size_t globalSizeBlur[2] = { (size_t)m_width, (size_t)m_height };
    size_t globalSizeSubtract = m_nPixels;
    
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_blurOpenclKernelRGBA,
        2,
        nullptr,
        globalSizeBlur,
        nullptr,
        0, nullptr, &blurEvent);

    clSetKernelArg(m_subtractOpenclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_subtractOpenclKernelRGBA, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_subtractOpenclKernelRGBA, 2, sizeof(int), &m_nPixels);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_subtractOpenclKernelRGBA,
        1,
        nullptr,
        &globalSizeSubtract,
        nullptr,
        0, nullptr, &subtractEvent);

    clWaitForEvents(1, &subtractEvent);

    clReleaseEvent(blurEvent);
    clReleaseEvent(subtractEvent);
}

void TrueOutlinesProcessor::init() {
    if (TrueOutlinesProcessor::firstTime) {
        if (isCudaAvailable()) {
            TrueOutlinesProcessor::processFunc = &TrueOutlinesProcessor::processCUDA;
            TrueOutlinesProcessor::processFuncRGBA = &TrueOutlinesProcessor::processRGBA_CUDA;

            TrueOutlinesProcessor::uploadFunc = &TrueOutlinesProcessor::uploadCUDA;
        }
        else {
            TrueOutlinesProcessor::processFunc = &TrueOutlinesProcessor::processOpenCL;
            TrueOutlinesProcessor::processFuncRGBA = &TrueOutlinesProcessor::processRGBA_OpenCL;

            TrueOutlinesProcessor::uploadFunc = &TrueOutlinesProcessor::uploadOpenCL;

            TrueOutlinesProcessor::m_blurOpenclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, blurOpenCLKernelSource, "blur_kernel");
            TrueOutlinesProcessor::m_blurOpenclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, blurRGBAOpenCLKernelSource, "blurRGBA_kernel");

            TrueOutlinesProcessor::m_subtractOpenclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, subtractOpenCLKernelSource, "subtract_kernel");
            TrueOutlinesProcessor::m_subtractOpenclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, subtractRGBAOpenCLKernelSource, "subtractRGBA_kernel");
        }
        TrueOutlinesProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
