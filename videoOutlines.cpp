#include "videoOutlines.h"
#include "outline_launcher.cuh"
#include "videoHeaders.h"


VOutlineWorker::VOutlineWorker(int shiftX, int shiftY, QObject* parent)
	: VideoEffect(parent), m_shiftX(shiftX), m_shiftY(shiftY)
{}

void VOutlineWorker::process() {
    try {
        // Generate temporary file names
        std::wstring current_time = std::to_wstring(std::time(nullptr));
        std::wstring video_root = fileUtils::splitextw(m_inputPath).first;
        std::wstring output_ext = fileUtils::splitextw(m_outputPath).second;

        std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
        std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";

        // Extract audio
        videoUtils::extractAudio(m_inputPath, temp_audio_name);

        Video video(m_inputPath, temp_video_name);
        Timer timer;

        unsigned char* d_img;
        unsigned char* d_img_copy;

        // Allocate device memory
        videoUtils::checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
        videoUtils::checkCudaError(cudaMalloc(&d_img_copy, video.getSize()), "Failed to allocate device memory for image copy");

        // Frame buffer pool (preallocated)
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
        videoUtils::checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");

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

            outlines(gridDim, blockDim, stream, d_img, d_img_copy, video.getWidth(), video.getHeight(), m_shiftX, m_shiftY);

            cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);

            timer.update();
            emit progressChanged(video, timer);
            video.nextFrame();

            cudaStreamSynchronize(stream);

            {
                std::lock_guard<std::mutex> frameLock(queueMutex);
                frameQueue.push(frameBuffer);
            }
            queueCV.notify_one();
        }

        isProcessing = false;
        queueCV.notify_one();
        writer.join();

        // clean up
        video.release();
        cudaFree(d_img);
        cudaFree(d_img_copy);
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
