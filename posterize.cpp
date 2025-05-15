#include "posterize.h"
#include "posterize_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (PosterizeProcessor::* PosterizeProcessor::processFunc)() const = nullptr;
void (PosterizeProcessor::* PosterizeProcessor::processFuncRGBA)() const = nullptr;

void (PosterizeProcessor::* PosterizeProcessor::uploadFunc)(const unsigned char*) = nullptr;
void (PosterizeProcessor::* PosterizeProcessor::downloadFunc)(unsigned char*) const = nullptr;

cl_kernel PosterizeProcessor::m_openclKernel = nullptr;
cl_kernel PosterizeProcessor::m_openclKernelRGBA = nullptr;

PosterizeProcessor::PosterizeProcessor(int size, float thresh)
    : imgSize(size), m_thresh(thresh) {

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

PosterizeProcessor::~PosterizeProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
    }
}

void PosterizeProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void PosterizeProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void PosterizeProcessor::upload(const unsigned char* Src) {
    (this->*uploadFunc)(Src);
}

void PosterizeProcessor::download(unsigned char* Dst) const {
    (this->*downloadFunc)(Dst);
}

void PosterizeProcessor::uploadCUDA(const unsigned char* Src) {
    cudaMemcpyAsync(d_img, Src, imgSize, cudaMemcpyHostToDevice, m_cudaStream);
}

void PosterizeProcessor::downloadCUDA(unsigned char* Dst) const {
    cudaMemcpyAsync(Dst, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void PosterizeProcessor::uploadOpenCL(const unsigned char* Src) {
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        Src,
        0, nullptr, nullptr);
}

void PosterizeProcessor::downloadOpenCL(unsigned char* Dst) const {
    clEnqueueReadBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_TRUE,  // Blocking read
        0,
        imgSize,
        Dst,
        0, nullptr, nullptr);
}

void PosterizeProcessor::process() const {
    (this->*processFunc)();
}

void PosterizeProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void PosterizeProcessor::processCUDA() const {
    posterize_CUDA(gridSize, blockSize, m_cudaStream, d_img, imgSize, m_thresh);
}

void PosterizeProcessor::processRGBA_CUDA() const {
    posterizeRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, imgSize, m_thresh);
}

void PosterizeProcessor::processOpenCL() const {
    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(int), &imgSize);
    clSetKernelArg(m_openclKernel, 2, sizeof(float), &m_thresh);

    size_t globalSize = imgSize;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernel,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void PosterizeProcessor::processRGBA_OpenCL() const {
    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(int), &imgSize);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(float), &m_thresh);

    size_t globalSize = imgSize;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernelRGBA,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &kernelEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void PosterizeProcessor::init() {
    if (isCudaAvailable()) {
        PosterizeProcessor::processFunc = &PosterizeProcessor::processCUDA;
        PosterizeProcessor::processFuncRGBA = &PosterizeProcessor::processRGBA_CUDA;

        PosterizeProcessor::uploadFunc = &PosterizeProcessor::uploadCUDA;
        PosterizeProcessor::downloadFunc = &PosterizeProcessor::downloadCUDA;
    }
    else {
        PosterizeProcessor::processFunc = &PosterizeProcessor::processOpenCL;
        PosterizeProcessor::processFuncRGBA = &PosterizeProcessor::processRGBA_OpenCL;

        PosterizeProcessor::uploadFunc = &PosterizeProcessor::uploadOpenCL;
        PosterizeProcessor::downloadFunc = &PosterizeProcessor::downloadOpenCL;

        PosterizeProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, posterizeOpenCLKernelSource, "posterize_kernel");
        PosterizeProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, posterizeRGBAOpenCLKernelSource, "posterizeRGBA_kernel");
    }
}
