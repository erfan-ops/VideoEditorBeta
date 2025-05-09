#include "videoVintage8bit.h"
#include "vintage8bit_launcher.cuh"
#include "videoHeaders.h"


void VVintage8bitWorker::process() {
    try {
        std::wstring current_time = std::to_wstring(std::time(nullptr));

        std::wstring video_root = fileUtils::splitextw(m_inputPath).first;
        std::wstring output_ext = fileUtils::splitextw(m_outputPath).second;

        std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
        std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

        videoUtils::extractAudio(m_inputPath, temp_audio_name);

        Video video(m_inputPath, temp_video_name);
        Timer timer;

        unsigned char* d_img;
        unsigned char* d_colorsBGR;
        unsigned char colorsBGR[] = {
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


        videoUtils::checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
        videoUtils::checkCudaError(cudaMalloc(&d_colorsBGR, sizeof(colorsBGR)), "Failed to allocate device memory for image");

        std::queue<cv::Mat> bufferPool;
        for (int i = 0; i < videoUtils::nBuffers; i++) {
            cv::Mat frame(video.getHeight(), video.getWidth(), video.getType());
            bufferPool.push(frame);
        }

        std::queue<cv::Mat> frameQueue;
        std::mutex queueMutex;
        std::condition_variable queueCV;
        std::atomic<bool> isProcessing(true);

        std::mutex bufferMutex;
        std::condition_variable bufferCV;

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
        videoUtils::checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

        cudaMemcpyAsync(d_colorsBGR, colorsBGR, sizeof(colorsBGR), cudaMemcpyHostToDevice, stream);

        dim3 blockDim(32, 32);
        dim3 gridDim((video.getWidth() + blockDim.x - 1) / blockDim.x, (video.getHeight() + blockDim.y - 1) / blockDim.y);

        int blockSize = 1024;
        int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;
        int roundGridSize = (video.getSize() + blockSize - 1) / blockSize;

        std::thread writer(writerThread);

        timer.start();
        while (video.getSuccess()) {
            std::unique_lock<std::mutex> bufferLock(bufferMutex);
            bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

            cv::Mat frameBuffer = bufferPool.front();
            bufferPool.pop();
            bufferLock.unlock();

            cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);

            vintage8bit(
                gridDim, blockDim, gridSize, blockSize, roundGridSize, stream,
                d_img, m_pixelWidth, m_pixelHeight, m_thresh, d_colorsBGR, sizeof(colorsBGR) / 3ULL,
                video.getWidth(), video.getHeight(), video.getNumPixels(), video.getSize()
            );

            cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            {
                std::lock_guard<std::mutex> frameLock(queueMutex);
                frameQueue.push(frameBuffer);
            }
            queueCV.notify_one();

            timer.update();
            emit progressChanged(video, timer);
            video.nextFrame();
        }

        isProcessing = false;
        queueCV.notify_one();
        writer.join();

        video.release();
        cudaFree(d_img);
        cudaFree(d_colorsBGR);
        cudaStreamDestroy(stream);

        videoUtils::mergeAudio(temp_video_name, temp_audio_name, m_outputPath);

        fileUtils::deleteFile(temp_video_name);
        fileUtils::deleteFile(temp_audio_name);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
