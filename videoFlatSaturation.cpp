#include "flatSaturation.h"
#include "videoFlatSaturation.h"
#include "videoHeaders.h"


void VFlatSaturationWorker::process() {
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

        FlatSaturationProcessor processor(video.getSize(), video.getNumPixels(), m_saturation);

        std::thread writer(writerThread);

        timer.start();
        while (video.getSuccess()) {
            std::unique_lock<std::mutex> bufferLock(bufferMutex);
            bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });

            cv::Mat frameBuffer = bufferPool.front();
            bufferPool.pop();
            bufferLock.unlock();

            processor.upload(video.getData());
            processor.process();
            processor.download(frameBuffer.data);

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

        // clean up
        video.release();

        videoUtils::mergeAudio(temp_video_name, temp_audio_name, m_outputPath);

        fileUtils::deleteFile(temp_video_name);
        fileUtils::deleteFile(temp_audio_name);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
