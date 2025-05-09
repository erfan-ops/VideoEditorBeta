#include "flatLight.h"
#include "flatLight_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (FlatLightProcessor::* FlatLightProcessor::processFunc)() const = nullptr;
void (FlatLightProcessor::* FlatLightProcessor::processFuncRGBA)() const = nullptr;

cl_kernel FlatLightProcessor::m_openclKernel = nullptr;
cl_kernel FlatLightProcessor::m_openclKernelRGBA = nullptr;

FlatLightProcessor::FlatLightProcessor(int size, int nPixels, float lightness)
    : imgSize(size), m_nPixels(nPixels), m_lightness(lightness) {
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

FlatLightProcessor::~FlatLightProcessor() {
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

void FlatLightProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void FlatLightProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void FlatLightProcessor::process() const {
    (this->*processFunc)();
}

void FlatLightProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void FlatLightProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    flatLight_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_lightness);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void FlatLightProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    flatLightRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_lightness);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void FlatLightProcessor::processOpenCL() const {
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
    clSetKernelArg(m_openclKernel, 2, sizeof(float), &m_lightness);

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

void FlatLightProcessor::processRGBA_OpenCL() const {
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
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(float), &m_lightness);

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

void FlatLightProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void FlatLightProcessor::upload(unsigned char* Dst) {
    memcpy(Dst, m_img, imgSize);
}

void FlatLightProcessor::init() {
    if (isCudaAvailable()) {
        FlatLightProcessor::processFunc = &FlatLightProcessor::processCUDA;
        FlatLightProcessor::processFuncRGBA = &FlatLightProcessor::processRGBA_CUDA;
    }
    else {
        FlatLightProcessor::processFunc = &FlatLightProcessor::processOpenCL;
        FlatLightProcessor::processFuncRGBA = &FlatLightProcessor::processRGBA_OpenCL;

        FlatLightProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, flatLightOpenCLKernelSource, "flatLight_kernel");
        FlatLightProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, flatLightRGBAOpenCLKernelSource, "flatLightRGBA_kernel");
    }
}
