#include "monoChrome.h"
#include "monoChrome_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (MonoChromeProcessor::* MonoChromeProcessor::processFunc)() const = nullptr;
void (MonoChromeProcessor::* MonoChromeProcessor::processFuncRGBA)() const = nullptr;

cl_kernel MonoChromeProcessor::m_openclKernel = nullptr;
cl_kernel MonoChromeProcessor::m_openclKernelRGBA = nullptr;

MonoChromeProcessor::MonoChromeProcessor(int size, int nPixels)
    : imgSize(size), m_nPixels(nPixels) {
    m_img = new unsigned char[this->imgSize];

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();
        blockSize = 1024;
        gridSize = (m_nPixels + blockSize - 1) / blockSize;
    }
    else {
        allocateOpenCL();
    }
}

MonoChromeProcessor::~MonoChromeProcessor() {
    // Free image and color palette buffers
    delete[] m_img;

    // Free CUDA buffers
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
    }
}

void MonoChromeProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void MonoChromeProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void MonoChromeProcessor::process() const {
    (this->*processFunc)();
}

void MonoChromeProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void MonoChromeProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    monoChrome_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void MonoChromeProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    monoChromeRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void MonoChromeProcessor::processOpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(int), &m_nPixels);

    size_t globalSize = m_nPixels;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernel,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // 4. Read back results (wait for kernel to complete)
    clEnqueueReadBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_TRUE,  // Blocking read
        0,
        imgSize,
        m_img,
        1, &kernelEvent, nullptr);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void MonoChromeProcessor::processRGBA_OpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(int), &m_nPixels);

    size_t globalSize = m_nPixels;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernelRGBA,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // 4. Read back results (wait for kernel to complete)
    clEnqueueReadBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_TRUE,  // Blocking read
        0,
        imgSize,
        m_img,
        1, &kernelEvent, nullptr);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void MonoChromeProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void MonoChromeProcessor::upload(unsigned char* Dst) const {
    memcpy(Dst, m_img, imgSize);
}

void MonoChromeProcessor::init() {
    if (isCudaAvailable()) {
        MonoChromeProcessor::processFunc = &MonoChromeProcessor::processCUDA;
        MonoChromeProcessor::processFuncRGBA = &MonoChromeProcessor::processRGBA_CUDA;
    }
    else {
        MonoChromeProcessor::processFunc = &MonoChromeProcessor::processOpenCL;
        MonoChromeProcessor::processFuncRGBA = &MonoChromeProcessor::processRGBA_OpenCL;

        MonoChromeProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, monoChromeOpenCLKernelSource, "monoChrome_kernel");
        MonoChromeProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, monoChromeRGBAOpenCLKernelSource, "monoChromeRGBA_kernel");
    }
}
