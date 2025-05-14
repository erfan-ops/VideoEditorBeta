#include "inverseContrast.h"
#include "inverseContrast_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (InverseContrastProcessor::* InverseContrastProcessor::processFunc)() const = nullptr;
void (InverseContrastProcessor::* InverseContrastProcessor::processFuncRGBA)() const = nullptr;

cl_kernel InverseContrastProcessor::m_openclKernel = nullptr;
cl_kernel InverseContrastProcessor::m_openclKernelRGBA = nullptr;

InverseContrastProcessor::InverseContrastProcessor(int size, int nPixels)
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

InverseContrastProcessor::~InverseContrastProcessor() {
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

void InverseContrastProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void InverseContrastProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void InverseContrastProcessor::process() const {
    (this->*processFunc)();
}

void InverseContrastProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void InverseContrastProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    inverseContrast_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void InverseContrastProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    inverseContrastRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void InverseContrastProcessor::processOpenCL() const {
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

void InverseContrastProcessor::processRGBA_OpenCL() const {
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

void InverseContrastProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void InverseContrastProcessor::upload(unsigned char* Dst) const {
    memcpy(Dst, m_img, imgSize);
}

void InverseContrastProcessor::init() {
    if (isCudaAvailable()) {
        InverseContrastProcessor::processFunc = &InverseContrastProcessor::processCUDA;
        InverseContrastProcessor::processFuncRGBA = &InverseContrastProcessor::processRGBA_CUDA;
    }
    else {
        InverseContrastProcessor::processFunc = &InverseContrastProcessor::processOpenCL;
        InverseContrastProcessor::processFuncRGBA = &InverseContrastProcessor::processRGBA_OpenCL;

        InverseContrastProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, inverseContrastOpenCLKernelSource, "inverseContrast_kernel");
        InverseContrastProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, inverseContrastRGBAOpenCLKernelSource, "inverseContrastRGBA_kernel");
    }
}
