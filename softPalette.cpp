#include "SoftPalette.h"
#include "softPalette_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

void (SoftPaletteProcessor::* SoftPaletteProcessor::processFunc)() const = nullptr;
void (SoftPaletteProcessor::* SoftPaletteProcessor::processFuncRGBA)() const = nullptr;

cl_kernel SoftPaletteProcessor::m_openclKernel = nullptr;
cl_kernel SoftPaletteProcessor::m_openclKernelRGBA = nullptr;

SoftPaletteProcessor::SoftPaletteProcessor(int nPixels, int size, unsigned char* colorsBGR, int numColors)
    : m_nPixels(nPixels), m_numColors(numColors) {
    this->imgSize = size;

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

SoftPaletteProcessor::~SoftPaletteProcessor() {
    if (isCudaAvailable()) {
        // Free CUDA buffers
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

void SoftPaletteProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_colorsBGR, 3ULL * m_numColors * sizeof(unsigned char), m_cudaStream);
}

void SoftPaletteProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_colorBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY,
        3ULL * m_numColors * sizeof(unsigned char), nullptr, &err);
}

void SoftPaletteProcessor::process() const {
    (this->*processFunc)();
}

void SoftPaletteProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void SoftPaletteProcessor::processCUDA() const {
    softPaletteLauncher(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, d_colorsBGR, m_numColors);
}

void SoftPaletteProcessor::processRGBA_CUDA() const {
    softPaletteLauncherRGBA(gridSize, blockSize, m_cudaStream, d_img, m_nPixels, d_colorsBGR, m_numColors);
}

void SoftPaletteProcessor::processOpenCL() const {
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

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void SoftPaletteProcessor::processRGBA_OpenCL() const {
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

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &kernelEvent);

    // Release the kernel event
    clReleaseEvent(kernelEvent);
}

void SoftPaletteProcessor::init() {
    if (isCudaAvailable()) {
        SoftPaletteProcessor::processFunc = &SoftPaletteProcessor::processCUDA;
        SoftPaletteProcessor::processFuncRGBA = &SoftPaletteProcessor::processRGBA_CUDA;
    }
    else {
        SoftPaletteProcessor::processFunc = &SoftPaletteProcessor::processOpenCL;
        SoftPaletteProcessor::processFuncRGBA = &SoftPaletteProcessor::processRGBA_OpenCL;

        SoftPaletteProcessor::m_openclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, softPaletteOpenCLKernelSource, "blendNearestColors_kernel");
        SoftPaletteProcessor::m_openclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, softPaletteRGBAOpenCLKernelSource, "blendNearestColorsRGBA_kernel");
    }
}
