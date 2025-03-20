#pragma once

#include "videoEditor.cuh"
#include "videoEffects.cuh"
#include "utils.h"

#include <Windows.h>
#include <filesystem>


constexpr int nBuffers = 8;


static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


static void extractAudio(const std::wstring& inputVideo, const std::wstring& outputAudio) {
    std::wstring audio_command = L"ffmpeg -loglevel quiet -i \"" + inputVideo + L"\" -vn -acodec copy \"" + outputAudio + L"\"";
    execute_command(audio_command);
}

static void mergeAudio(const std::wstring& inputVideo, const std::wstring& inputAudio, const std::wstring& outputVideo) {
    std::wstring merge_command = L"ffmpeg -loglevel quiet -i \"" + inputVideo + L"\" -i \"" + inputAudio + L"\" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 \"" + outputVideo + L"\" -y";
    execute_command(merge_command);
}


__host__ void videoVintage8bit(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight,
    const unsigned char* colors_BGR,
    const size_t nColors,
    const unsigned char threshold,
    const unsigned short lineWidth,
    const unsigned char lineDarkeningThresh
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;
    unsigned char* d_colors_BGR;

    size_t color_size = 3ULL * nColors * sizeof(unsigned char);

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_colors_BGR, color_size), "Failed to allocate device memory for colors");

    checkCudaError(cudaMemcpy(d_colors_BGR, colors_BGR, color_size, cudaMemcpyHostToDevice), "Failed to copy colors to device");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    dim3 blockDim(32, 32);
    dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

    int flatBlockSize = 1024;
    int flatGridSize = (video.getNumPixels() + flatBlockSize - 1) / flatBlockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        dynamicColor_kernel<<<flatGridSize, flatBlockSize, 0, stream>>>(d_img, video.getNumPixels(), d_colors_BGR, nColors);
        censor_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), pixelWidth, pixelHeight);
        roundColors_kernel<<<flatGridSize, flatBlockSize, 0, stream>>>(d_img, video.getNumPixels(), threshold);
        horizontalLine_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), lineWidth, lineDarkeningThresh);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    video.release();
    cudaFree(d_img);
    cudaFree(d_colors_BGR);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}


__host__ void videoRadialBlur(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    int blurRadius,
    float intensity,
    float centerX,
    float centerY
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    if (centerX == -1)
        centerX = video.getWidth() / 2.0f;
    if (centerY == -1)
        centerY = video.getHeight() / 2.0f;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    dim3 blockDim(32, 32);
    dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        radial_blur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), centerX, centerY, blurRadius, intensity);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}


__host__ void videoReverseContrast(
    const std::wstring& inputPath,
    const std::wstring& outputPath
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        reverse_contrast<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels());

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}


__host__ void videoShiftHue(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    float hue_shift
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    float rotationFactor = 2.0f * hue_shift;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        shift_hue_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels(), rotationFactor);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoOutlines(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    int shiftX, int shiftY
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;
    unsigned char* d_img_copy;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_img_copy, video.getSize()), "Failed to allocate device memory for image copy");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    dim3 blockDim(32, 32);
    dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_img_copy, d_img, video.getSize(), cudaMemcpyDeviceToDevice, stream);

        // fix intelisense
        outlines_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, video.getHeight(), video.getWidth(), shiftX, shiftY);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaFree(d_img_copy);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoHighlightMotion(
    const std::wstring& inputPath,
    const std::wstring& outputPath
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_oldImg;
    unsigned char* d_newImg;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_oldImg, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_newImg, video.getSize()), "Failed to allocate device memory for image copy");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    cudaMemcpyAsync(d_newImg, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);
    video.nextFrame();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_oldImg, d_newImg, video.getSize(), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_newImg, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        subtract_kernel<<<gridSize, blockSize, 0, stream>>>(d_oldImg, d_newImg, video.getNumPixels());

        cudaMemcpyAsync(frameBuffer.data, d_oldImg, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_oldImg);
    cudaFree(d_newImg);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoBlur(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const int blurRadius,
    const int blending,
    const float precision
) {

    using KernelFunction = void (*)(unsigned char* __restrict__ img, const unsigned char* __restrict__ img_copy, const int rows, const int cols, const int blur_radius);
    KernelFunction blur_func = nullptr;
    if (blending == 0) {
        blur_func = &fastBlur_kernel;
    }
    else if (blending == 1) {
        blur_func = &trueBlur_kernel;
    }
    else {
        blur_func = &fastBlur_kernel;
    }

    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;
    unsigned char* d_img_copy;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_img_copy, video.getSize()), "Failed to allocate device memory for image copy");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    dim3 blockDim(32, 32);
    dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

    std::thread writer(writerThread);

    timer.start();
    if (blending == 2) {
        while (video.getSuccess()) {
            std::unique_lock<std::mutex> bufferLock(bufferMutex);
            bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

            cv::Mat frameBuffer = bufferPool.front();
            bufferPool.pop();
            bufferLock.unlock();

            cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_img_copy, d_img, video.getSize(), cudaMemcpyDeviceToDevice, stream);

            // fix intelisense
            preciseBlur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, video.getHeight(), video.getWidth(), blurRadius, precision);

            cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            {
                std::lock_guard<std::mutex> frameLock(queueMutex);
                frameQueue.push(frameBuffer);
            }
            queueCV.notify_one();

            timer.update();
            videoShowProgress(video, timer);
            video.nextFrame();
        }
    }
    else {
        while (video.getSuccess()) {
            std::unique_lock<std::mutex> bufferLock(bufferMutex);
            bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

            cv::Mat frameBuffer = bufferPool.front();
            bufferPool.pop();
            bufferLock.unlock();

            cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_img_copy, d_img, video.getSize(), cudaMemcpyDeviceToDevice, stream);

            // fix intelisense
            trueBlur_kernel << <gridDim, blockDim, 0, stream >> > (d_img, d_img_copy, video.getHeight(), video.getWidth(), blurRadius);

            cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            {
                std::lock_guard<std::mutex> frameLock(queueMutex);
                frameQueue.push(frameBuffer);
            }
            queueCV.notify_one();

            timer.update();
            videoShowProgress(video, timer);
            video.nextFrame();
        }
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaFree(d_img_copy);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoTrueOutlines(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const int thresh
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;
    unsigned char* d_img_copy;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_img_copy, video.getSize()), "Failed to allocate device memory for image copy");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    dim3 blockDim(32, 32);
    dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_img_copy, d_img, video.getSize(), cudaMemcpyDeviceToDevice, stream);

        // fix intelisense
        fastBlur_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, d_img_copy, video.getHeight(), video.getWidth(), thresh);
        subtract_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, d_img_copy, video.getNumPixels());

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaFree(d_img_copy);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoMonoChrome(
    const std::wstring& inputPath,
    const std::wstring& outputPath
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        monoChrome_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels());

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoMonoMask(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char* colors_BGR,
    const size_t nColors
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;
    unsigned char* d_colors_BGR;

    size_t color_size = 3ULL * nColors * sizeof(unsigned char);

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_colors_BGR, color_size), "Failed to allocate device memory for colors");

    checkCudaError(cudaMemcpy(d_colors_BGR, colors_BGR, color_size, cudaMemcpyHostToDevice), "Failed to copy colors to device");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        dynamicColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels(), d_colors_BGR, nColors);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoPassColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const float* __restrict passThreshValues
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;
    float* d_passThreshValues;

    static constexpr size_t color_size = 3ULL * sizeof(float);

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_passThreshValues, color_size), "Failed to allocate device memory for colors");

    checkCudaError(cudaMemcpy(d_passThreshValues, passThreshValues, color_size, cudaMemcpyHostToDevice), "Failed to copy colors to device");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        passColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels(), d_passThreshValues);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoPixelate(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    dim3 blockDim(32, 32);
    dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        pixelate_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), pixelWidth, pixelHeight);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoCensor(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short pixelWidth,
    const unsigned short pixelHeight
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    dim3 blockDim(32, 32);
    dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        censor_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), pixelWidth, pixelHeight);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoRoundColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned char thresh
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        roundColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels(), thresh);

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}

__host__ void videoInverseColors(
    const std::wstring& inputPath,
    const std::wstring& outputPath
) {
    // Generate temporary file names
    std::wstring current_time = std::to_wstring(std::time(nullptr));

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

    // Extract audio
    extractAudio(inputPath, temp_audio_name);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = nBuffers;
    std::queue<cv::Mat> bufferPool;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        cv::Mat frame(video.getImage().size(), video.getImage().type());
        bufferPool.push(frame);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> isProcessing(true);

    std::mutex bufferMutex;
    std::condition_variable bufferCV;

    // Writer thread function
    auto writerThread = [&]() {
        while (true) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });

                if (!isProcessing && frameQueue.empty()) break;

                frame = frameQueue.front();
                frameQueue.pop();
            }

            video.write(frame);

            // Recycle buffer
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                bufferPool.push(frame);
                bufferCV.notify_one();
            }
        }
        };

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

    int blockSize = 1024;
    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;

    std::thread writer(writerThread);

    timer.start();
    while (video.getSuccess()) {
        std::unique_lock<std::mutex> bufferLock(bufferMutex);
        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

        cv::Mat frameBuffer = bufferPool.front();
        bufferPool.pop();
        bufferLock.unlock();

        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

        // fix intelisense
        inverseColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels());

        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        {
            std::lock_guard<std::mutex> frameLock(queueMutex);
            frameQueue.push(frameBuffer);
        }
        queueCV.notify_one();

        timer.update();
        videoShowProgress(video, timer);
        video.nextFrame();
    }

    isProcessing = false;
    queueCV.notify_one();
    writer.join();

    // clean up
    video.release();
    cudaFree(d_img);
    cudaStreamDestroy(stream);

    mergeAudio(temp_video_name, temp_audio_name, outputPath);

    fileUtils::deleteFile(temp_video_name);
    fileUtils::deleteFile(temp_audio_name);
}
