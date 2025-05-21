#include "pixelate.h"
#include "pixelate_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

bool PixelateProcessor::firstTime = true;

void (PixelateProcessor::* PixelateProcessor::processFunc)() const = nullptr;
void (PixelateProcessor::* PixelateProcessor::processFuncRGBA)() const = nullptr;

cl_kernel PixelateProcessor::m_openclKernel = nullptr;
cl_kernel PixelateProcessor::m_openclKernelRGBA = nullptr;

PixelateProcessor::PixelateProcessor(int size, int width, int height, int pixelWidth, int pixelHeight)
    : m_width(width), m_height(height), m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight) {
    imgSize = size;

    if (PixelateProcessor::firstTime) {
        PixelateProcessor::init();
        PixelateProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
            BaseProcessor::firstTime = false;
        }
    }

    this->m_xBound = (m_width + m_pixelWidth - 1) / m_pixelWidth;
    this->m_yBound = (m_height + m_pixelHeight - 1) / m_pixelHeight;
    
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

PixelateProcessor::~PixelateProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
    }
}

void PixelateProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void PixelateProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void PixelateProcessor::process() const {
    (this->*processFunc)();
}

void PixelateProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void PixelateProcessor::processCUDA() const {
    pixelate_CUDA(gridDim, blockDim, m_cudaStream, d_img, m_width, m_height, m_pixelWidth, m_pixelHeight, m_xBound, m_yBound);
}

void PixelateProcessor::processRGBA_CUDA() const {
    pixelateRGBA_CUDA(gridDim, blockDim, m_cudaStream, d_img, m_width, m_height, m_pixelWidth, m_pixelHeight, m_xBound, m_yBound);
}

void PixelateProcessor::processOpenCL() const {
    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernel, 2, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernel, 3, sizeof(int), &m_pixelWidth);
    clSetKernelArg(m_openclKernel, 4, sizeof(int), &m_pixelHeight);
    clSetKernelArg(m_openclKernel, 5, sizeof(int), &m_xBound);
    clSetKernelArg(m_openclKernel, 6, sizeof(int), &m_yBound);

    size_t globalWorkSize[2] = { (size_t)m_xBound, (size_t)m_yBound };
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

void PixelateProcessor::processRGBA_OpenCL() const {
    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernelRGBA, 3, sizeof(int), &m_pixelWidth);
    clSetKernelArg(m_openclKernelRGBA, 4, sizeof(int), &m_pixelHeight);
    clSetKernelArg(m_openclKernelRGBA, 5, sizeof(int), &m_xBound);
    clSetKernelArg(m_openclKernelRGBA, 6, sizeof(int), &m_yBound);

    size_t globalWorkSize[2] = { (size_t)m_xBound, (size_t)m_yBound };
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

void PixelateProcessor::init() {
    if (PixelateProcessor::firstTime) {
        if (isCudaAvailable()) {
            PixelateProcessor::processFunc = &PixelateProcessor::processCUDA;
            PixelateProcessor::processFuncRGBA = &PixelateProcessor::processRGBA_CUDA;
        }
        else {
            PixelateProcessor::processFunc = &PixelateProcessor::processOpenCL;
            PixelateProcessor::processFuncRGBA = &PixelateProcessor::processRGBA_OpenCL;

            PixelateProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, pixelateOpenCLKernelSource, "pixelate_kernel");
            PixelateProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, pixelateRGBAOpenCLKernelSource, "pixelateRGBA_kernel");
        }
        PixelateProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
