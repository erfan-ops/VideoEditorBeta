#include "Blur.h"
#include "blur_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (BlurProcessor::* BlurProcessor::processFunc)() const = nullptr;
void (BlurProcessor::* BlurProcessor::processFuncRGBA)() const = nullptr;

cl_kernel BlurProcessor::m_openclKernel = nullptr;
cl_kernel BlurProcessor::m_openclKernelRGBA = nullptr;

BlurProcessor::BlurProcessor(int size, int width, int height, int radius)
    : imgSize(size), m_width(width), m_height(height), m_radius(radius) {
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

BlurProcessor::~BlurProcessor() {
    // Free image and color palette buffers
    delete[] m_img;

    // Free CUDA buffers
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaFree(d_img_copy);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
        clReleaseMemObject(m_imgCopyBuf);
    }
}

void BlurProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_img_copy, imgSize, m_cudaStream);
}

void BlurProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);
    m_imgCopyBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY, imgSize, nullptr, &err);
}

void BlurProcessor::process() const {
    (this->*processFunc)();
}

void BlurProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void BlurProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);
    cudaMemcpyAsync(d_img_copy, d_img, imgSize, cudaMemcpyDeviceToDevice, m_cudaStream);

    blur_CUDA(gridDim, blockDim, m_cudaStream, d_img, d_img_copy, m_width, m_height, m_radius);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void BlurProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);
    cudaMemcpyAsync(d_img_copy, d_img, imgSize, cudaMemcpyDeviceToDevice, m_cudaStream);

    blurRGBA_CUDA(gridDim, blockDim, m_cudaStream, d_img, d_img_copy, m_width, m_height, m_radius);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void BlurProcessor::processOpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);
    clEnqueueCopyBuffer(globalQueueOpenCL,
        m_imgBuf,        // src buffer
        m_imgCopyBuf,    // dst buffer
        0,               // src offset
        0,               // dst offset
        imgSize,         // size to copy
        0, nullptr, nullptr);


    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_openclKernel, 2, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernel, 3, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernel, 4, sizeof(int), &m_radius);

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

void BlurProcessor::processRGBA_OpenCL() const {
    // 1. Copy input image to device
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,  // Non-blocking write
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);
    clEnqueueCopyBuffer(globalQueueOpenCL,
        m_imgBuf,        // src buffer
        m_imgCopyBuf,    // dst buffer
        0,               // src offset
        0,               // dst offset
        imgSize,         // size to copy
        0, nullptr, nullptr);


    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(cl_mem), &m_imgCopyBuf);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(int), &m_height);
    clSetKernelArg(m_openclKernelRGBA, 3, sizeof(int), &m_width);
    clSetKernelArg(m_openclKernelRGBA, 4, sizeof(int), &m_radius);

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

void BlurProcessor::setImage(unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

unsigned char* BlurProcessor::getImage() {
    return m_img;
}

void BlurProcessor::init() {
    if (isCudaAvailable()) {
        BlurProcessor::processFunc = &BlurProcessor::processCUDA;
        BlurProcessor::processFuncRGBA = &BlurProcessor::processRGBA_CUDA;
    }
    else {
        BlurProcessor::processFunc = &BlurProcessor::processOpenCL;
        BlurProcessor::processFuncRGBA = &BlurProcessor::processRGBA_OpenCL;

        BlurProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, blurOpenCLKernelSource, "blur_kernel");
        BlurProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, blurRGBAOpenCLKernelSource, "blurRGBA_kernel");
    }
}
