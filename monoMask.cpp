#include "monoMask.h"
#include "monoMask_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (MonoMaskProcessor::* MonoMaskProcessor::processFunc)() const = nullptr;
void (MonoMaskProcessor::* MonoMaskProcessor::processFuncRGBA)() const = nullptr;

cl_kernel MonoMaskProcessor::m_openclKernel = nullptr;
cl_kernel MonoMaskProcessor::m_openclKernelRGBA = nullptr;

MonoMaskProcessor::MonoMaskProcessor(int nPixels, int size, unsigned char* colorsBGR, int numColors)
    : m_nPixels(nPixels), m_numColors(numColors), imgSize(size) {
    // Allocate memory for image and color palette
    m_img = new unsigned char[this->imgSize];  // For BGR (3 channels)

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&m_cudaStream);
        allocateCUDA();
        blockSize = 1024;
        gridSize = (m_nPixels + blockSize - 1) / blockSize;
        cudaMemcpyAsync(d_colorsBGR, colorsBGR, 3ULL * m_numColors * sizeof(unsigned char), cudaMemcpyHostToDevice, m_cudaStream);
    }
    else {
        allocateOpenCL();
        clEnqueueWriteBuffer(globalQueueOpenCL,
            m_colorBuf,
            CL_TRUE,  // Blocking write
            0,
            3ULL * m_numColors * sizeof(unsigned char),
            colorsBGR,
            0, nullptr, nullptr);
    }
}

MonoMaskProcessor::~MonoMaskProcessor() {
    // Free image and color palette buffers
    delete[] m_img;

    // Free CUDA buffers
    if (isCudaAvailable()) {
        if (d_img) {
            cudaFree(d_img);
        }
        if (d_colorsBGR) {
            cudaFree(d_colorsBGR);
        }
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        // Free OpenCL buffers
        if (m_imgBuf) {
            clReleaseMemObject(m_imgBuf);
        }
        if (m_colorBuf) {
            clReleaseMemObject(m_colorBuf);
        }
    }
}

void MonoMaskProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_colorsBGR, 3ULL * m_numColors * sizeof(unsigned char), m_cudaStream);
}

void MonoMaskProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_colorBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY,
        3ULL * m_numColors * sizeof(unsigned char), nullptr, &err);
}

void MonoMaskProcessor::process() const {
    (this->*processFunc)();
}

void MonoMaskProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void MonoMaskProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    monoMask_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, d_colorsBGR, m_numColors);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void MonoMaskProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    monoMaskRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, d_colorsBGR, m_numColors);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void MonoMaskProcessor::processOpenCL() const {
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
    clSetKernelArg(m_openclKernel, 2, sizeof(cl_mem), &m_colorBuf);
    clSetKernelArg(m_openclKernel, 3, sizeof(int), &m_numColors);

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

void MonoMaskProcessor::processRGBA_OpenCL() const {
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
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(cl_mem), &m_colorBuf);
    clSetKernelArg(m_openclKernelRGBA, 3, sizeof(int), &m_numColors);

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

void MonoMaskProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void MonoMaskProcessor::upload(unsigned char* Dst) const {
    memcpy(Dst, m_img, imgSize);
}

void MonoMaskProcessor::init() {
    if (isCudaAvailable()) {
        MonoMaskProcessor::processFunc = &MonoMaskProcessor::processCUDA;
        MonoMaskProcessor::processFuncRGBA = &MonoMaskProcessor::processRGBA_CUDA;
    }
    else {
        MonoMaskProcessor::processFunc = &MonoMaskProcessor::processOpenCL;
        MonoMaskProcessor::processFuncRGBA = &MonoMaskProcessor::processRGBA_OpenCL;

        MonoMaskProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, monoMaskOpenCLKernelSource, "monoMask_kernel");
        MonoMaskProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, monoMaskRGBAOpenCLKernelSource, "monoMaskRGBA_kernel");
    }
}
