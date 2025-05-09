#include "censor.h"
#include "censor_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (CensorProcessor::* CensorProcessor::processFunc)() const = nullptr;
void (CensorProcessor::* CensorProcessor::processFuncRGBA)() const = nullptr;

cl_kernel CensorProcessor::m_openclKernel = nullptr;
cl_kernel CensorProcessor::m_openclKernelRGBA = nullptr;

CensorProcessor::CensorProcessor(int size, int width, int height, int pixelWidth, int pixelHeight)
    : imgSize(size), m_width(width), m_height(height), m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight) {
    m_img = new unsigned char[this->imgSize];

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();
        blockDim = dim3(32, 32);
        gridDim = dim3((m_width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    }
    else {
        allocateOpenCL();
    }
}

CensorProcessor::~CensorProcessor() {
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

void CensorProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void CensorProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void CensorProcessor::process() const {
    (this->*processFunc)();
}

void CensorProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void CensorProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    censor_CUDA(gridDim, blockDim, m_cudaStream, d_img, m_width, m_height, m_pixelWidth, m_pixelHeight);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void CensorProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    censorRGBA_CUDA(gridDim, blockDim, m_cudaStream, d_img, m_width, m_height, m_pixelWidth, m_pixelHeight);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void CensorProcessor::processOpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernel, 2, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernel, 3, sizeof(int), &m_pixelWidth);
    clSetKernelArg(m_openclKernel, 4, sizeof(int), &m_pixelHeight);

    size_t globalWorkSize[2] = { (size_t)m_width, (size_t)m_height };
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernel,
        2,
        nullptr,
        globalWorkSize,
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

void CensorProcessor::processRGBA_OpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernelRGBA, 3, sizeof(int), &m_pixelWidth);
    clSetKernelArg(m_openclKernelRGBA, 4, sizeof(int), &m_pixelHeight);

    size_t globalWorkSize[2] = { (size_t)m_width, (size_t)m_height };
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernelRGBA,
        2,
        nullptr,
        globalWorkSize,
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

void CensorProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void CensorProcessor::upload(unsigned char* Dst) {
    memcpy(Dst, m_img, imgSize);
}

void CensorProcessor::init() {
    if (isCudaAvailable()) {
        CensorProcessor::processFunc = &CensorProcessor::processCUDA;
        CensorProcessor::processFuncRGBA = &CensorProcessor::processRGBA_CUDA;
    }
    else {
        CensorProcessor::processFunc = &CensorProcessor::processOpenCL;
        CensorProcessor::processFuncRGBA = &CensorProcessor::processRGBA_OpenCL;

        CensorProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, censorOpenCLKernelSource, "censor_kernel");
        CensorProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, censorRGBAOpenCLKernelSource, "censorRGBA_kernel");
    }
}
