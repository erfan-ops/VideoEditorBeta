#include "radialBlur.h"
#include "radialBlur_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

bool RadialBlurProcessor::firstTime = true;

void (RadialBlurProcessor::* RadialBlurProcessor::uploadFunc)(const unsigned char* Src) = nullptr;

void (RadialBlurProcessor::* RadialBlurProcessor::processFunc)() const = nullptr;
void (RadialBlurProcessor::* RadialBlurProcessor::processFuncRGBA)() const = nullptr;

cl_kernel RadialBlurProcessor::m_openclKernel = nullptr;
cl_kernel RadialBlurProcessor::m_openclKernelRGBA = nullptr;

RadialBlurProcessor::RadialBlurProcessor(int size, int width, int height, float centerX, float centerY, int blurRadius, float intensity)
    : m_width(width), m_height(height), m_centerX(centerX), m_centerY(centerY), m_blurRadius(blurRadius), m_intensity(intensity) {
    imgSize = size;

    if (RadialBlurProcessor::firstTime) {
        RadialBlurProcessor::init();
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
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

RadialBlurProcessor::~RadialBlurProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaFree(d_imgCopy);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
        clReleaseMemObject(m_imgCopyBuf);
    }
}

void RadialBlurProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_imgCopy, imgSize, m_cudaStream);
}

void RadialBlurProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_imgCopyBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY, imgSize, nullptr, &err);
}

void RadialBlurProcessor::uploadCUDA(const unsigned char* Src) {
    cudaMemcpyAsync(d_img, Src, imgSize, cudaMemcpyHostToDevice, m_cudaStream);
    cudaMemcpyAsync(d_imgCopy, d_img, imgSize, cudaMemcpyDeviceToDevice, m_cudaStream);
}

void RadialBlurProcessor::uploadOpenCL(const unsigned char* Src) {
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

void RadialBlurProcessor::process() const {
    (this->*processFunc)();
}

void RadialBlurProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void RadialBlurProcessor::processCUDA() const {
    radialBlur_CUDA(
        gridDim, blockDim, m_cudaStream,
        d_img, d_imgCopy, m_width, m_height,
        m_centerX, m_centerY, m_blurRadius, m_intensity
    );
}

void RadialBlurProcessor::processRGBA_CUDA() const {
    radialBlurRGBA_CUDA(
        gridDim, blockDim, m_cudaStream,
        d_img, d_imgCopy, m_width, m_height,
        m_centerX, m_centerY, m_blurRadius, m_intensity
    );
}

void RadialBlurProcessor::processOpenCL() const {
    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_openclKernel, 2, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernel, 3, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernel, 4, sizeof(float), &m_centerX);
    clSetKernelArg(m_openclKernel, 5, sizeof(float), &m_centerY);
    clSetKernelArg(m_openclKernel, 6, sizeof(int), &m_blurRadius);
    clSetKernelArg(m_openclKernel, 7, sizeof(float), &m_intensity);

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

void RadialBlurProcessor::processRGBA_OpenCL() const {
    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernelRGBA, 3, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernelRGBA, 4, sizeof(float), &m_centerX);
    clSetKernelArg(m_openclKernelRGBA, 5, sizeof(float), &m_centerY);
    clSetKernelArg(m_openclKernelRGBA, 6, sizeof(int), &m_blurRadius);
    clSetKernelArg(m_openclKernelRGBA, 7, sizeof(float), &m_intensity);

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

void RadialBlurProcessor::init() {
    if (RadialBlurProcessor::firstTime) {
        if (isCudaAvailable()) {
            RadialBlurProcessor::processFunc = &RadialBlurProcessor::processCUDA;
            RadialBlurProcessor::processFuncRGBA = &RadialBlurProcessor::processRGBA_CUDA;

            RadialBlurProcessor::uploadFunc = &RadialBlurProcessor::uploadCUDA;
        }
        else {
            RadialBlurProcessor::processFunc = &RadialBlurProcessor::processOpenCL;
            RadialBlurProcessor::processFuncRGBA = &RadialBlurProcessor::processRGBA_OpenCL;

            RadialBlurProcessor::uploadFunc = &RadialBlurProcessor::uploadOpenCL;

            RadialBlurProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, radialBlurOpenCLKernelSource, "radialBlur_kernel");
            RadialBlurProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, radialBlurRGBAOpenCLKernelSource, "radialBlurRGBA_kernel");
        }
        RadialBlurProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
