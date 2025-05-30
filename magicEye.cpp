#include "magicEye.h"
#include "magicEye_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"


bool MagicEyeProcessor::firstTime = true;

void (MagicEyeProcessor::* MagicEyeProcessor::processFunc)() const = nullptr;

cl_kernel MagicEyeProcessor::m_subtractKernel = nullptr;
cl_kernel MagicEyeProcessor::m_noiseKernel = nullptr;
cl_kernel MagicEyeProcessor::m_bwKernel = nullptr;

MagicEyeProcessor::MagicEyeProcessor(int size, int nPixels, float middle)
    : m_nPixels(nPixels), m_middle(middle) {
    if (MagicEyeProcessor::firstTime) {
        MagicEyeProcessor::init();
        MagicEyeProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
            BaseProcessor::firstTime = false;
        }
    }

    this->imgSize = size;

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();
        blockSize = 1024;
        gridSize = (m_nPixels + blockSize - 1) / blockSize;
        binaryNoise_CUDA(gridSize, blockSize, this->m_cudaStream, d_noise, m_nPixels, time(nullptr));
        cudaStreamSynchronize(this->m_cudaStream);
    }
    else {
        globalSize = m_nPixels;
        allocateOpenCL();

        unsigned int seed = time(nullptr) | 0xffffffffU;

        clSetKernelArg(m_noiseKernel, 0, sizeof(cl_mem), &m_noiseBuf);
        clSetKernelArg(m_noiseKernel, 1, sizeof(int), &m_nPixels);
        clSetKernelArg(m_noiseKernel, 2, sizeof(unsigned int), &seed);

        cl_event kernelEvent;
        clEnqueueNDRangeKernel(globalQueueOpenCL,
            m_noiseKernel,
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
}

MagicEyeProcessor::~MagicEyeProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaFree(d_noise);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
        clReleaseMemObject(m_noiseBuf);
    }
}

void MagicEyeProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_noise, imgSize, m_cudaStream);
}

void MagicEyeProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_noiseBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY, imgSize, nullptr, &err);
}

void MagicEyeProcessor::process() const {
    (this->*processFunc)();
}

void MagicEyeProcessor::processCUDA() const {
    magicEye_CUDA(gridSize, blockSize, m_cudaStream, d_img, d_noise, m_nPixels, m_middle);
}

void MagicEyeProcessor::processOpenCL() const {
    clSetKernelArg(m_bwKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_bwKernel, 1, sizeof(int), &m_nPixels);
    clSetKernelArg(m_bwKernel, 2, sizeof(float), &m_middle);

    cl_event event1, event2;
    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_bwKernel,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &event1);

    clSetKernelArg(m_subtractKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_subtractKernel, 1, sizeof(cl_mem), &m_noiseBuf);
    clSetKernelArg(m_subtractKernel, 2, sizeof(int), &m_nPixels);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_subtractKernel,
        1,
        nullptr,
        &globalSize,
        nullptr,
        0, nullptr, &event2);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &event2);

    // Release kernel events
    clReleaseEvent(event1);
    clReleaseEvent(event2);
}

void MagicEyeProcessor::init() {
    if (MagicEyeProcessor::firstTime) {
        if (isCudaAvailable()) {
            MagicEyeProcessor::processFunc = &MagicEyeProcessor::processCUDA;
        }
        else {
            MagicEyeProcessor::processFunc = &MagicEyeProcessor::processOpenCL;

            MagicEyeProcessor::m_subtractKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, subtractOpenCLKernelSource, "subtract_kernel");
            MagicEyeProcessor::m_noiseKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, binaryNoiseOpenCLKernelSource, "binaryNoise_kernel");
            MagicEyeProcessor::m_bwKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, blackAndWhiteOpenCLKernelSource, "blackAndWhite_kernel");
        }
        MagicEyeProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
