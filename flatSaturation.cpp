#include "flatSaturation.h"
#include "flatSaturation_CUDA.cuh"
#include "flatSaturation_OpenCL.h"
#include "globals.h"
#include "utils.h"

bool FlatSaturationProcessor::firstTime = true;

void (FlatSaturationProcessor::* FlatSaturationProcessor::processFunc)() const = nullptr;
void (FlatSaturationProcessor::* FlatSaturationProcessor::processFuncRGBA)() const = nullptr;

cl_kernel FlatSaturationProcessor::m_openclKernel = nullptr;
cl_kernel FlatSaturationProcessor::m_openclKernelRGBA = nullptr;

FlatSaturationProcessor::FlatSaturationProcessor(int size, int nPixels, float thresh)
    : m_thresh(thresh), m_nPixels(nPixels) {
    imgSize = size;

    if (FlatSaturationProcessor::firstTime) {
        FlatSaturationProcessor::init();
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
    }
    else {
        globalWorkSize = m_nPixels;
        allocateOpenCL();
    }
}

FlatSaturationProcessor::~FlatSaturationProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
    }
}

void FlatSaturationProcessor::process() const {
    (this->*processFunc)();
}

void FlatSaturationProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void FlatSaturationProcessor::processCUDA() const {
    flatSaturation_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_thresh);
}

void FlatSaturationProcessor::processRGBA_CUDA() const {
    flatSaturationRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_thresh);
}

void FlatSaturationProcessor::processOpenCL() const {
    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(int), &m_nPixels);
    clSetKernelArg(m_openclKernel, 2, sizeof(float), &m_thresh);

    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernel,
        1,
        nullptr,
        &globalWorkSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void FlatSaturationProcessor::processRGBA_OpenCL() const {
    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(int), &m_nPixels);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(float), &m_thresh);

    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernelRGBA,
        1,
        nullptr,
        &globalWorkSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void FlatSaturationProcessor::init() {
    if (FlatSaturationProcessor::firstTime) {
        if (isCudaAvailable()) {
            FlatSaturationProcessor::processFunc = &FlatSaturationProcessor::processCUDA;
            FlatSaturationProcessor::processFuncRGBA = &FlatSaturationProcessor::processRGBA_CUDA;
        }
        else {
            FlatSaturationProcessor::processFunc = &FlatSaturationProcessor::processOpenCL;
            FlatSaturationProcessor::processFuncRGBA = &FlatSaturationProcessor::processRGBA_OpenCL;

            FlatSaturationProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, flatSaturationOpenCLKernelSource, "flatSaturation_kernel");
            FlatSaturationProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, flatSaturationRGBAOpenCLKernelSource, "flatSaturationRGBA_kernel");
        }
        FlatSaturationProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
