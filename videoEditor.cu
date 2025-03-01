#pragma once

#include "videoEditor.cuh"
#include "videoEffects.cuh"
#include "utils.h"

#include <Windows.h>


static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


__host__ void videoVintage8bit(
    const std::wstring& inputPath,
    const std::wstring& outputPath,
    const unsigned short& pixelWidth,
    const unsigned short& pixelHeight,
    const unsigned char* color1,
    const unsigned char* color2,
    const unsigned char* color3,
    const unsigned char& threshold,
    const unsigned short& lineWidth,
    const unsigned char& lineDarkeningThresh
) {
    // Generate temporary file names
    std::time_t current_time = std::time(nullptr);
    std::wstring time_string = stringUtils::string_to_wstring(std::ctime(&current_time));
    time_string.erase(std::remove(time_string.begin(), time_string.end(), ':'), time_string.end());
    time_string.erase(time_string.find_last_not_of('\n') + 1);

    std::wstring video_root = fileUtils::splitextw(inputPath).first;
    std::wstring output_ext = fileUtils::splitextw(outputPath).second;

    std::wstring temp_video_name = video_root + L" " + time_string + output_ext;
    std::wstring temp_audio_name = video_root + L" " + time_string + L".aac";

    // Extract audio
    std::wstring audio_command = L"ffmpeg -loglevel quiet -i \"" + inputPath + L"\" -vn -acodec copy \"" + temp_audio_name + L"\"";
    execute_command(audio_command);

    Video video(inputPath, temp_video_name);
    Timer timer;

    unsigned char* d_img;
    unsigned char* d_color1;
    unsigned char* d_color2;
    unsigned char* d_color3;

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_color1, 3 * sizeof(unsigned char)), "Failed to allocate device memory for color1");
    checkCudaError(cudaMalloc(&d_color2, 3 * sizeof(unsigned char)), "Failed to allocate device memory for color2");
    checkCudaError(cudaMalloc(&d_color3, 3 * sizeof(unsigned char)), "Failed to allocate device memory for color3");

    checkCudaError(cudaMemcpy(d_color1, color1, 3 * sizeof(unsigned char), cudaMemcpyHostToDevice), "Failed to copy color to device");
    checkCudaError(cudaMemcpy(d_color2, color2, 3 * sizeof(unsigned char), cudaMemcpyHostToDevice), "Failed to copy color to device");
    checkCudaError(cudaMemcpy(d_color3, color3, 3 * sizeof(unsigned char), cudaMemcpyHostToDevice), "Failed to copy color to device");

    // Frame buffer pool (preallocated)
    const int NUM_BUFFERS = 4;
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
        triColor_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), d_color1, d_color2, d_color3);
        censor_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), pixelWidth, pixelHeight);
        roundColors_kernel<<<gridDim, blockDim, 0, stream>>>(d_img, video.getHeight(), video.getWidth(), threshold);
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
    cudaFree(d_color1);
    cudaFree(d_color2);
    cudaFree(d_color3);
    cudaStreamDestroy(stream);

    std::wstring merge_command = L"ffmpeg -loglevel quiet -i \"" + temp_video_name + L"\" -i \"" + temp_audio_name + L"\" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 \"" + outputPath + L"\" -y";
    execute_command(merge_command);

    execute_command(L"del \"" + temp_video_name + L"\"");
    execute_command(L"del \"" + temp_audio_name + L"\"");
}
