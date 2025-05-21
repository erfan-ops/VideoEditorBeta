#include "posterize.h"
#include "posterize_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

bool PosterizeProcessor::firstTime = true;

void (PosterizeProcessor::* PosterizeProcessor::processFunc)() const = nullptr;
void (PosterizeProcessor::* PosterizeProcessor::processFuncRGBA)() const = nullptr;

cl_kernel PosterizeProcessor::m_openclKernel = nullptr;
cl_kernel PosterizeProcessor::m_openclKernelRGBA = nullptr;

PosterizeProcessor::PosterizeProcessor(int size, float thresh)
    : m_thresh(thresh) {
	if (PosterizeProcessor::firstTime) {
        PosterizeProcessor::init();
        PosterizeProcessor::firstTime = false;
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
    if (PosterizeProcessor::firstTime) {
        if (isCudaAvailable()) {
            PosterizeProcessor::processFunc = &PosterizeProcessor::processCUDA;
            PosterizeProcessor::processFuncRGBA = &PosterizeProcessor::processRGBA_CUDA;
        }
        else {
            PosterizeProcessor::processFunc = &PosterizeProcessor::processOpenCL;
            PosterizeProcessor::processFuncRGBA = &PosterizeProcessor::processRGBA_OpenCL;

            PosterizeProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, posterizeOpenCLKernelSource, "posterize_kernel");
            PosterizeProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, posterizeRGBAOpenCLKernelSource, "posterizeRGBA_kernel");
        }
        PosterizeProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
