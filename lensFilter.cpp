#include "lensFilter.h"
#include "lensFilter_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (LensFilterProcessor::* LensFilterProcessor::processFunc)() const = nullptr;
void (LensFilterProcessor::* LensFilterProcessor::processFuncRGBA)() const = nullptr;

cl_kernel LensFilterProcessor::m_openclKernel = nullptr;
cl_kernel LensFilterProcessor::m_openclKernelRGBA = nullptr;

LensFilterProcessor::LensFilterProcessor(int size, float* passThreshValues, int nThreshValues)
    : imgSize(size), m_nThreshValues(nThreshValues) {
    m_img = new unsigned char[this->imgSize];

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();
        blockSize = 1024;
        gridSize = (imgSize + blockSize - 1) / blockSize;
        cudaMemcpyAsync(d_passThreshValues, passThreshValues, m_nThreshValues * sizeof(float), cudaMemcpyHostToDevice, m_cudaStream);
    }
    else {
        allocateOpenCL();
        clEnqueueWriteBuffer(globalQueueOpenCL,
            m_passThreshValuesBuf,
            CL_TRUE,  // Blocking write
            0,
            m_nThreshValues * sizeof(float),
            passThreshValues,
            0, nullptr, nullptr);
    }
}

LensFilterProcessor::~LensFilterProcessor() {
    // Free image and color palette buffers
    delete[] m_img;

    // Free CUDA buffers
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaFree(d_passThreshValues);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
        clReleaseMemObject(m_passThreshValuesBuf);
    }
}

void LensFilterProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_passThreshValues, m_nThreshValues * sizeof(float), m_cudaStream);
}

void LensFilterProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_passThreshValuesBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY, m_nThreshValues * sizeof(float), nullptr, &err);
}

void LensFilterProcessor::process() const {
    (this->*processFunc)();
}

void LensFilterProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void LensFilterProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    lensFilter_CUDA(gridSize, blockSize, m_cudaStream, d_img, imgSize, d_passThreshValues);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void LensFilterProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    lensFilterRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, imgSize, d_passThreshValues);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void LensFilterProcessor::processOpenCL() const {
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
    clSetKernelArg(m_openclKernel, 2, sizeof(cl_mem), &m_passThreshValuesBuf);

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

void LensFilterProcessor::processRGBA_OpenCL() const {
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
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(cl_mem), &m_passThreshValuesBuf);

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

void LensFilterProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void LensFilterProcessor::upload(unsigned char* Dst) const {
    memcpy(Dst, m_img, imgSize);
}

void LensFilterProcessor::init() {
    if (isCudaAvailable()) {
        LensFilterProcessor::processFunc = &LensFilterProcessor::processCUDA;
        LensFilterProcessor::processFuncRGBA = &LensFilterProcessor::processRGBA_CUDA;
    }
    else {
        LensFilterProcessor::processFunc = &LensFilterProcessor::processOpenCL;
        LensFilterProcessor::processFuncRGBA = &LensFilterProcessor::processRGBA_OpenCL;

        LensFilterProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, lensFilterOpenCLKernelSource, "lensFilter_kernel");
        LensFilterProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, lensFilterRGBAOpenCLKernelSource, "lensFilterRGBA_kernel");
    }
}
