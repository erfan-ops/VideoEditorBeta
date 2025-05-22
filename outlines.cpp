#include "outlines.h"
#include "outline_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

bool OutlinesProcessor::firstTime = true;

void (OutlinesProcessor::* OutlinesProcessor::uploadFunc)(const unsigned char* Src) = nullptr;
void (OutlinesProcessor::* OutlinesProcessor::downloadFunc)(unsigned char* Dst) const = nullptr;

void (OutlinesProcessor::* OutlinesProcessor::processFunc)() const = nullptr;
void (OutlinesProcessor::* OutlinesProcessor::processFuncRGBA)() const = nullptr;

cl_kernel OutlinesProcessor::m_openclKernel = nullptr;
cl_kernel OutlinesProcessor::m_openclKernelRGBA = nullptr;

OutlinesProcessor::OutlinesProcessor(int size, int width, int height, int xShift, int yShift)
    : m_width(width), m_height(height), m_xShift(xShift), m_yShift(yShift) {
    imgSize = size;

    if (OutlinesProcessor::firstTime) {
        OutlinesProcessor::init();
        OutlinesProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
            BaseProcessor::firstTime = false;
        }
    }

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();

        blockDim = dim3(32, 32);
        gridDim = dim3(
            (m_width + blockDim.x - 1) / blockDim.x,
            (m_height + blockDim.y - 1) / blockDim.y
        );
    }
    else {
        allocateOpenCL();
    }
}

OutlinesProcessor::~OutlinesProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
    }
}

void OutlinesProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_imgCopy, imgSize, m_cudaStream);
}

void OutlinesProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_imgCopyBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void OutlinesProcessor::uploadCUDA(const unsigned char* Src) {
    cudaMemcpyAsync(d_img, Src, imgSize, cudaMemcpyHostToDevice, m_cudaStream);
    cudaMemcpyAsync(d_imgCopy, d_img, imgSize, cudaMemcpyDeviceToDevice, m_cudaStream);
}

void OutlinesProcessor::uploadOpenCL(const unsigned char* Src) {
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,
        0,
        imgSize,
        Src,
        0, nullptr, nullptr);
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgCopyBuf,
        CL_FALSE,
        0,
        imgSize,
        Src,
        0, nullptr, nullptr);
}

void OutlinesProcessor::process() const {
    (this->*processFunc)();
}

void OutlinesProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void OutlinesProcessor::processCUDA() const {
    outlines_CUDA(gridDim, blockDim, m_cudaStream, d_img, d_imgCopy, m_width, m_height, m_xShift, m_yShift);
}

void OutlinesProcessor::processRGBA_CUDA() const {
    outlinesRGBA_CUDA(gridDim, blockDim, m_cudaStream, d_img, d_imgCopy, m_width, m_height, m_xShift, m_yShift);
}

void OutlinesProcessor::processOpenCL() const {
    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_openclKernel, 2, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernel, 3, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernel, 4, sizeof(int), &m_xShift);
    clSetKernelArg(m_openclKernel, 5, sizeof(int), &m_yShift);

    size_t globalWorkSize[2] = { (size_t)m_width, (size_t)m_height };
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernel,
        2,
        nullptr,
        globalWorkSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void OutlinesProcessor::processRGBA_OpenCL() const {
    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernelRGBA, 3, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernelRGBA, 4, sizeof(int), &m_xShift);
    clSetKernelArg(m_openclKernelRGBA, 5, sizeof(int), &m_yShift);

    size_t globalWorkSize[2] = { (size_t)m_width, (size_t)m_height };
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernelRGBA,
        2,
        nullptr,
        globalWorkSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void OutlinesProcessor::init() {
    if (OutlinesProcessor::firstTime) {
        if (isCudaAvailable()) {
            OutlinesProcessor::processFunc = &OutlinesProcessor::processCUDA;
            OutlinesProcessor::processFuncRGBA = &OutlinesProcessor::processRGBA_CUDA;

            OutlinesProcessor::uploadFunc = &OutlinesProcessor::uploadCUDA;
            OutlinesProcessor::downloadFunc = &OutlinesProcessor::downloadCUDA;
        }
        else {
            OutlinesProcessor::processFunc = &OutlinesProcessor::processOpenCL;
            OutlinesProcessor::processFuncRGBA = &OutlinesProcessor::processRGBA_OpenCL;

            OutlinesProcessor::uploadFunc = &OutlinesProcessor::uploadOpenCL;
            OutlinesProcessor::downloadFunc = &OutlinesProcessor::downloadOpenCL;

            OutlinesProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, outlinesOpenCLKernelSource, "outlines_kernel");
            OutlinesProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, outlinesRGBAOpenCLKernelSource, "outlinesRGBA_kernel");
        }
        OutlinesProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
