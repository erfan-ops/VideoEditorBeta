#include "blackAndWhite.h"
#include "blackAndWhite_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (BlackAndWhiteProcessor::* BlackAndWhiteProcessor::processFunc)() const = nullptr;
void (BlackAndWhiteProcessor::* BlackAndWhiteProcessor::processFuncRGBA)() const = nullptr;

cl_kernel BlackAndWhiteProcessor::m_openclKernel = nullptr;
cl_kernel BlackAndWhiteProcessor::m_openclKernelRGBA = nullptr;

BlackAndWhiteProcessor::BlackAndWhiteProcessor(int nPixels, int size, float middle)
    : m_nPixels(nPixels), imgSize(size), m_middle(middle) {
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

BlackAndWhiteProcessor::~BlackAndWhiteProcessor() {
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

void BlackAndWhiteProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void BlackAndWhiteProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void BlackAndWhiteProcessor::process() const {
    (this->*processFunc)();
}

void BlackAndWhiteProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void BlackAndWhiteProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    blackAndWhite_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_middle);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void BlackAndWhiteProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    blackAndWhiteRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_middle);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void BlackAndWhiteProcessor::processOpenCL() const {
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
    clSetKernelArg(m_openclKernel, 2, sizeof(float), &m_middle);

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

void BlackAndWhiteProcessor::processRGBA_OpenCL() const {
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
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(float), &m_middle);

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

void BlackAndWhiteProcessor::setImage(unsigned char* img, int size) {
    memcpy(m_img, img, size);
}

unsigned char* BlackAndWhiteProcessor::getImage() {
    return m_img;
}

void BlackAndWhiteProcessor::init() {
    if (isCudaAvailable()) {
        BlackAndWhiteProcessor::processFunc = &BlackAndWhiteProcessor::processCUDA;
        BlackAndWhiteProcessor::processFuncRGBA = &BlackAndWhiteProcessor::processRGBA_CUDA;
    }
    else {
        BlackAndWhiteProcessor::processFunc = &BlackAndWhiteProcessor::processOpenCL;
        BlackAndWhiteProcessor::processFuncRGBA = &BlackAndWhiteProcessor::processRGBA_OpenCL;

        BlackAndWhiteProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, blackAndWhiteOpenCLKernelSource, "blackAndWhite_kernel");
        BlackAndWhiteProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, blackAndWhiteRGBAOpenCLKernelSource, "blackAndWhiteRGBA_kernel");
    }
}
