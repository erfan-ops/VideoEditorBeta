#include "baseProcessor.h"

bool BaseProcessor::firstTime = true;

void (BaseProcessor::* BaseProcessor::uploadFunc)(const unsigned char*) = nullptr;
void (BaseProcessor::* BaseProcessor::downloadFunc)(unsigned char*) const = nullptr;

void BaseProcessor::allocateCUDA() {
	cudaMallocAsync(&d_img, imgSize, m_cudaStream);
}

void BaseProcessor::allocateOpenCL() {
	cl_int err;
	m_imgBuf = clCreateBuffer(globalContextOpenCL, CL_MEM_READ_WRITE, imgSize, nullptr, &err);
}

void BaseProcessor::uploadCUDA(const unsigned char* Src) {
	cudaMemcpyAsync(d_img, Src, imgSize, cudaMemcpyHostToDevice, m_cudaStream);
}

void BaseProcessor::uploadOpenCL(const unsigned char* Src) {
	clEnqueueWriteBuffer(globalQueueOpenCL,
		m_imgBuf,
		CL_FALSE,
		0,
		imgSize,
		Src,
		0, nullptr, nullptr);
}

void BaseProcessor::downloadCUDA(unsigned char* Dst) const {
	cudaMemcpyAsync(Dst, d_img, imgSize, cudaMemcpyDeviceToHost, m_cudaStream);
	cudaStreamSynchronize(m_cudaStream);
}

void BaseProcessor::downloadOpenCL(unsigned char* Dst) const {
	clEnqueueReadBuffer(globalQueueOpenCL,
		m_imgBuf,
		CL_TRUE,  // Blocking read
		0,
		imgSize,
		Dst,
		0, nullptr, nullptr);
}

void BaseProcessor::init() {
	if (BaseProcessor::firstTime) {
		if (isCudaAvailable()) {
			BaseProcessor::uploadFunc = &BaseProcessor::uploadCUDA;
			BaseProcessor::downloadFunc = &BaseProcessor::downloadCUDA;
		}
		else {
			BaseProcessor::uploadFunc = &BaseProcessor::uploadOpenCL;
			BaseProcessor::downloadFunc = &BaseProcessor::downloadOpenCL;
		}
		BaseProcessor::firstTime = false;
	}
}
