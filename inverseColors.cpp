#include "inverseColors.h"
#include "inverseColors_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (InverseColorsProcessor::* InverseColorsProcessor::processFunc)() const = nullptr;
void (InverseColorsProcessor::* InverseColorsProcessor::processFuncRGBA)() const = nullptr;

cl_kernel InverseColorsProcessor::m_openclKernel = nullptr;
cl_kernel InverseColorsProcessor::m_openclKernelRGBA = nullptr;

InverseColorsProcessor::InverseColorsProcessor(int size)
    : imgSize(size) {
    m_img = new unsigned char[this->imgSize];

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();
        blockSize = 1024;
        gridSize = (imgSize + blockSize - 1) / blockSize;
    }
    else {
        allocateOpenCL();
    }
}

InverseColorsProcessor::~InverseColorsProcessor() {
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

void InverseColorsProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void InverseColorsProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void InverseColorsProcessor::process() const {
    (this->*processFunc)();
}

void InverseColorsProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void InverseColorsProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    inverseColors_CUDA(gridSize, blockSize, m_cudaStream, d_img, imgSize);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void InverseColorsProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    inverseColorsRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, imgSize);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void InverseColorsProcessor::processOpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(int), &imgSize);

    size_t globalSize = imgSize;
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

void InverseColorsProcessor::processRGBA_OpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(int), &imgSize);

    size_t globalSize = imgSize;
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

void InverseColorsProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void InverseColorsProcessor::upload(unsigned char* Dst) const {
    memcpy(Dst, m_img, imgSize);
}

void InverseColorsProcessor::init() {
    if (isCudaAvailable()) {
        InverseColorsProcessor::processFunc = &InverseColorsProcessor::processCUDA;
        InverseColorsProcessor::processFuncRGBA = &InverseColorsProcessor::processRGBA_CUDA;
    }
    else {
        InverseColorsProcessor::processFunc = &InverseColorsProcessor::processOpenCL;
        InverseColorsProcessor::processFuncRGBA = &InverseColorsProcessor::processRGBA_OpenCL;

        InverseColorsProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, inverseColorsOpenCLKernelSource, "inverseColors_kernel");
        InverseColorsProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, inverseColorsRGBAOpenCLKernelSource, "inverseColorsRGBA_kernel");
    }
}
