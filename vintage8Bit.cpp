#include "vintage8Bit.h"
#include "vintage8Bit_launcher.cuh"
#include "globals.h"
#include "utils.h"
#include "SourcesOpenCL.h"

bool Vintage8BitProcessor::firstTime = true;

void (Vintage8BitProcessor::* Vintage8BitProcessor::processFunc)() const = nullptr;
void (Vintage8BitProcessor::* Vintage8BitProcessor::processFuncRGBA)() const = nullptr;

cl_kernel Vintage8BitProcessor::m_changePaletteOpenclKernel = nullptr;
cl_kernel Vintage8BitProcessor::m_changePaletteOpenclKernelRGBA = nullptr;
cl_kernel Vintage8BitProcessor::m_censorOpenclKernel = nullptr;
cl_kernel Vintage8BitProcessor::m_censorOpenclKernelRGBA = nullptr;
cl_kernel Vintage8BitProcessor::m_posterizeOpenclKernel = nullptr;
cl_kernel Vintage8BitProcessor::m_posterizeOpenclKernelRGBA = nullptr;

unsigned char Vintage8BitProcessor::colors_BGR[42] = {
    64, 9, 67,
    61, 70, 133,
    59, 131, 197,
    58, 124, 127,
    61, 64, 61,
    122, 188, 191,
    122, 194, 255,
    121, 246, 255,
    187, 251, 254,
    125, 134, 197,
    56, 72, 118,
    120, 202, 250,
    47, 126, 205,
    20, 44, 105
};

Vintage8BitProcessor::Vintage8BitProcessor(int size, int width, int height, int pixelWidth, int pixelHeight, float thresh)
    : m_width(width), m_height(height), m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight), m_nPixels(width* height), m_thresh(thresh) {
    imgSize = size;
    m_nColors = sizeof(colors_BGR) / sizeof(unsigned char) / 3;

    if (Vintage8BitProcessor::firstTime) {
        Vintage8BitProcessor::init();
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }

    // Allocate buffers for CUDA or OpenCL
    if (isCudaAvailable()) {
        cudaStreamCreate(&this->m_cudaStream);
        allocateCUDA();

        cudaMemcpyAsync(d_colors_BGR, colors_BGR, sizeof(colors_BGR), cudaMemcpyHostToDevice, this->m_cudaStream);

        blockSize = 1024;
        gridSize = (m_nPixels + blockSize - 1) / blockSize;
        roundGridSize = (imgSize + blockSize - 1) / blockSize;

        blockDim = dim3(32, 32);
        gridDim = dim3(
            (this->m_width + blockDim.x - 1) / blockDim.x,
            (this->m_height + blockDim.y - 1) / blockDim.y
        );
    }
    else {
        allocateOpenCL();
        clEnqueueWriteBuffer(globalQueueOpenCL,
            m_colorsBuf,
            CL_FALSE,
            0,
            sizeof(colors_BGR),
            colors_BGR,
            0, nullptr, nullptr);
    }
}

Vintage8BitProcessor::~Vintage8BitProcessor() {
    if (isCudaAvailable()) {
        cudaFree(d_img);
        cudaStreamDestroy(m_cudaStream);
    }
    else {
        clReleaseMemObject(m_imgBuf);
    }
}

void Vintage8BitProcessor::allocateCUDA() {
    cudaMallocAsync(&d_img, imgSize, m_cudaStream);
    cudaMallocAsync(&d_colors_BGR, sizeof(colors_BGR), m_cudaStream);
}

void Vintage8BitProcessor::allocateOpenCL() {
    cl_int err;
    m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
    m_colorsBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_ONLY, sizeof(colors_BGR), nullptr, &err);
}

void Vintage8BitProcessor::process() const {
    (this->*processFunc)();
}

void Vintage8BitProcessor::processRGBA() const {
    (this->*processFuncRGBA)();
}

void Vintage8BitProcessor::processCUDA() const {
    vintage8bit_CUDA(
        gridDim, blockDim, gridSize, blockSize, roundGridSize, m_cudaStream,
        d_img, m_pixelWidth, m_pixelHeight, m_thresh, d_colors_BGR, m_nColors,
        m_width, m_height, m_nPixels, imgSize
    );
}

void Vintage8BitProcessor::processRGBA_CUDA() const {
    vintage8bitRGBA_CUDA(
        gridDim, blockDim, gridSize, blockSize, roundGridSize, m_cudaStream,
        d_img, m_pixelWidth, m_pixelHeight, m_thresh, d_colors_BGR, m_nColors,
        m_width, m_height, m_nPixels, imgSize
    );
}

void Vintage8BitProcessor::processOpenCL() const {
    cl_event changePaletteEvent, censorEvent, posterizeEvent;

    size_t globalSizeChangePalette = m_nPixels;
    size_t globalSizeCensor[2] = { (size_t)m_width, (size_t)m_height };
    size_t globalSizePosterize = imgSize;

    clSetKernelArg(m_changePaletteOpenclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_changePaletteOpenclKernel, 1, sizeof(int), &m_nPixels);
    clSetKernelArg(m_changePaletteOpenclKernel, 2, sizeof(cl_mem), &m_colorsBuf);
    clSetKernelArg(m_changePaletteOpenclKernel, 3, sizeof(int), &m_nColors);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_changePaletteOpenclKernel,
        1,
        nullptr,
        &globalSizeChangePalette,
        nullptr,
        0, nullptr, &changePaletteEvent);

    clSetKernelArg(m_censorOpenclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_censorOpenclKernel, 1, sizeof(int), &m_height);
    clSetKernelArg(m_censorOpenclKernel, 2, sizeof(int), &m_width);
    clSetKernelArg(m_censorOpenclKernel, 3, sizeof(int), &m_pixelWidth);
    clSetKernelArg(m_censorOpenclKernel, 4, sizeof(int), &m_pixelHeight);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_censorOpenclKernel,
        2,
        nullptr,
        globalSizeCensor,
        nullptr,
        0, nullptr, &censorEvent);

    clSetKernelArg(m_posterizeOpenclKernel, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_posterizeOpenclKernel, 1, sizeof(int), &imgSize);
    clSetKernelArg(m_posterizeOpenclKernel, 2, sizeof(float), &m_thresh);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_posterizeOpenclKernel,
        1,
        nullptr,
        &globalSizePosterize,
        nullptr,
        0, nullptr, &posterizeEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &posterizeEvent);

    // Release the kernel event
    clReleaseEvent(changePaletteEvent);
    clReleaseEvent(censorEvent);
    clReleaseEvent(posterizeEvent);
}

void Vintage8BitProcessor::processRGBA_OpenCL() const {
    cl_event changePaletteEvent, censorEvent, posterizeEvent;

    size_t globalSizeChangePalette = m_nPixels;
    size_t globalSizeCensor[2] = { (size_t)m_width, (size_t)m_height };
    size_t globalSizePosterize = imgSize;

    clSetKernelArg(m_changePaletteOpenclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_changePaletteOpenclKernelRGBA, 1, sizeof(int), &m_nPixels);
    clSetKernelArg(m_changePaletteOpenclKernelRGBA, 2, sizeof(cl_mem), &m_colorsBuf);
    clSetKernelArg(m_changePaletteOpenclKernelRGBA, 3, sizeof(int), &m_nColors);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_changePaletteOpenclKernelRGBA,
        1,
        nullptr,
        &globalSizeChangePalette,
        nullptr,
        0, nullptr, &changePaletteEvent);

    clSetKernelArg(m_censorOpenclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_censorOpenclKernelRGBA, 1, sizeof(int), &m_height);
    clSetKernelArg(m_censorOpenclKernelRGBA, 2, sizeof(int), &m_width);
    clSetKernelArg(m_censorOpenclKernelRGBA, 3, sizeof(int), &m_pixelWidth);
    clSetKernelArg(m_censorOpenclKernelRGBA, 4, sizeof(int), &m_pixelHeight);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_censorOpenclKernelRGBA,
        2,
        nullptr,
        globalSizeCensor,
        nullptr,
        0, nullptr, &censorEvent);

    clSetKernelArg(m_posterizeOpenclKernelRGBA, 0, sizeof(cl_mem), &m_imgBuf);
    clSetKernelArg(m_posterizeOpenclKernelRGBA, 1, sizeof(int), &imgSize);
    clSetKernelArg(m_posterizeOpenclKernelRGBA, 2, sizeof(float), &m_thresh);

    clEnqueueNDRangeKernel(globalQueueOpenCL,
        m_posterizeOpenclKernelRGBA,
        1,
        nullptr,
        &globalSizePosterize,
        nullptr,
        0, nullptr, &posterizeEvent);

    // ⏳ Wait for kernel to complete
    clWaitForEvents(1, &posterizeEvent);

    // Release the kernel event
    clReleaseEvent(changePaletteEvent);
    clReleaseEvent(censorEvent);
    clReleaseEvent(posterizeEvent);
}

void Vintage8BitProcessor::init() {
    if (Vintage8BitProcessor::firstTime) {
        if (isCudaAvailable()) {
            Vintage8BitProcessor::processFunc = &Vintage8BitProcessor::processCUDA;
            Vintage8BitProcessor::processFuncRGBA = &Vintage8BitProcessor::processRGBA_CUDA;
        }
        else {
            Vintage8BitProcessor::processFunc = &Vintage8BitProcessor::processOpenCL;
            Vintage8BitProcessor::processFuncRGBA = &Vintage8BitProcessor::processRGBA_OpenCL;

            Vintage8BitProcessor::m_changePaletteOpenclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, changePaletteOpenCLKernelSource, "changePalette_kernel");
            Vintage8BitProcessor::m_changePaletteOpenclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, changePaletteRGBAOpenCLKernelSource, "changePaletteRGBA_kernel");

            Vintage8BitProcessor::m_censorOpenclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, censorOpenCLKernelSource, "censor_kernel");
            Vintage8BitProcessor::m_censorOpenclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, censorRGBAOpenCLKernelSource, "censorRGBA_kernel");

            Vintage8BitProcessor::m_posterizeOpenclKernel = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, posterizeOpenCLKernelSource, "posterize_kernel");
            Vintage8BitProcessor::m_posterizeOpenclKernelRGBA = openclUtils::createKernelFromSource(globalContextOpenCL, globalDeviceOpenCL, posterizeRGBAOpenCLKernelSource, "posterizeRGBA_kernel");
        }
        Vintage8BitProcessor::firstTime = false;
        if (BaseProcessor::firstTime) {
            BaseProcessor::init();
        }
    }
}
