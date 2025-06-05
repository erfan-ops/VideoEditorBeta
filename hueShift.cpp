#include "hueShift.h"
#include "hueShift_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (HueShiftProcessor::* HueShiftProcessor::processFunc)() const = nullptr;
void (HueShiftProcessor::* HueShiftProcessor::processFuncRGBA)() const = nullptr;

cl_kernel HueShiftProcessor::m_openclKernel = nullptr;
cl_kernel HueShiftProcessor::m_openclKernelRGBA = nullptr;

HueShiftProcessor::HueShiftProcessor(int size, int nPixels, float hue, float saturation, float lightness)
    : imgSize(size), m_nPixels(nPixels), m_hue(hue), m_saturation(saturation), m_lightness(lightness) {
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

HueShiftProcessor::~HueShiftProcessor() {
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

void HueShiftProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void HueShiftProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void HueShiftProcessor::process() const {
    (this->*processFunc)();
}

void HueShiftProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void HueShiftProcessor::processCUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    hueShift_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_hue, m_saturation, m_lightness);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void HueShiftProcessor::processRGBA_CUDA() const {
    cudaMemcpyAsync(d_img, m_img, imgSize, cudaMemcpyHostToDevice, m_cudaStream);

    hueShiftRGBA_CUDA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, m_hue, m_saturation, m_lightness);

    cudaMemcpyAsync(m_img, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);
}

void HueShiftProcessor::processOpenCL() const {
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernel, 1, sizeof(int), &m_nPixels);
    clSetKernelArg(m_openclKernel, 2, sizeof(float), &m_hue);
    clSetKernelArg(m_openclKernel, 3, sizeof(float), &m_saturation);
    clSetKernelArg(m_openclKernel, 4, sizeof(float), &m_lightness);

    size_t globalSize = m_nPixels;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernel,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &kernelEvent);

    clEnqueueReadBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_TRUE,
        0,
        imgSize,
        m_img,
        1, &kernelEvent, nullptr);

    clReleaseEvent(kernelEvent);
}

void HueShiftProcessor::processRGBA_OpenCL() const {
    clEnqueueWriteBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_FALSE,
        0,
        imgSize,
        m_img,
        0, nullptr, nullptr);

    clSetKernelArg(m_openclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_openclKernelRGBA, 1, sizeof(int), &m_nPixels);
    clSetKernelArg(m_openclKernelRGBA, 2, sizeof(float), &m_hue);
    clSetKernelArg(m_openclKernelRGBA, 3, sizeof(float), &m_saturation);
    clSetKernelArg(m_openclKernelRGBA, 4, sizeof(float), &m_lightness);

    size_t globalSize = m_nPixels;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_openclKernelRGBA,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &kernelEvent);

    clEnqueueReadBuffer(globalQueueOpenCL,
        m_imgBuf,
        CL_TRUE,
        0,
        imgSize,
        m_img,
        1, &kernelEvent, nullptr);

    clReleaseEvent(kernelEvent);
}

void HueShiftProcessor::setImage(const unsigned char* img) {
    memcpy(m_img, img, imgSize);
}

void HueShiftProcessor::upload(unsigned char* Dst) const {
    memcpy(Dst, m_img, imgSize);
}

void HueShiftProcessor::init() {
    if (isCudaAvailable()) {
        HueShiftProcessor::processFunc = &HueShiftProcessor::processCUDA;
        HueShiftProcessor::processFuncRGBA = &HueShiftProcessor::processRGBA_CUDA;
    }
    else {
        HueShiftProcessor::processFunc = &HueShiftProcessor::processOpenCL;
        HueShiftProcessor::processFuncRGBA = &HueShiftProcessor::processRGBA_OpenCL;

        HueShiftProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, hueShiftOpenCLKernelSource, "HSL_kernel");
        HueShiftProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, hueShiftRGBAOpenCLKernelSource, "HSLRGBA_kernel");
    }
}
