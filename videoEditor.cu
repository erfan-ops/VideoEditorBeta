//__host__ void videoMonoMask(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const unsigned char* colors_BGR,
//    const size_t nColors
//) {
//    // Generate temporary file names
//    std::wstring current_time = std::to_wstring(std::time(nullptr));
//
//    std::wstring video_root = fileUtils::splitextw(inputPath).first;
//    std::wstring output_ext = fileUtils::splitextw(outputPath).second;
//
//    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
//    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";
//
//    // Extract audio
//    extractAudio(inputPath, temp_audio_name);
//
//    Video video(inputPath, temp_video_name);
//    Timer timer;
//
//    unsigned char* d_img;
//    unsigned char* d_colors_BGR;
//
//    size_t color_size = 3ULL * nColors * sizeof(unsigned char);
//
//    // Allocate device memory
//    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
//    checkCudaError(cudaMalloc(&d_colors_BGR, color_size), "Failed to allocate device memory for colors");
//
//    checkCudaError(cudaMemcpy(d_colors_BGR, colors_BGR, color_size, cudaMemcpyHostToDevice), "Failed to copy colors to device");
//
//    // Frame buffer pool (preallocated)
//    const int NUM_BUFFERS = nBuffers;
//    std::queue<cv::Mat> bufferPool;
//    for (int i = 0; i < NUM_BUFFERS; i++) {
//        cv::Mat frame(video.getHeight(), video.getWidth(), video.getType());
//        bufferPool.push(frame);
//    }
//
//    std::queue<cv::Mat> frameQueue;
//    std::mutex queueMutex;
//    std::condition_variable queueCV;
//    std::atomic<bool> isProcessing(true);
//
//    std::mutex bufferMutex;
//    std::condition_variable bufferCV;
//
//    // Writer thread function
//    auto writerThread = [&]() {
//        while (true) {
//            cv::Mat frame;
//            {
//                std::unique_lock<std::mutex> lock(queueMutex);
//                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });
//
//                if (!isProcessing && frameQueue.empty()) break;
//
//                frame = frameQueue.front();
//                frameQueue.pop();
//            }
//
//            video.write(frame);
//
//            // Recycle buffer
//            {
//                std::lock_guard<std::mutex> bufferLock(bufferMutex);
//                bufferPool.push(frame);
//                bufferCV.notify_one();
//            }
//        }
//        };
//
//    cudaStream_t stream;
//    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");
//
//    int blockSize = 1024;
//    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;
//
//    std::thread writer(writerThread);
//
//    timer.start();
//    while (video.getSuccess()) {
//        std::unique_lock<std::mutex> bufferLock(bufferMutex);
//        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });
//
//        cv::Mat frameBuffer = bufferPool.front();
//        bufferPool.pop();
//        bufferLock.unlock();
//
//        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);
//
//        // fix intelisense
//        dynamicColor_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels(), d_colors_BGR, nColors);
//
//        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
//        cudaStreamSynchronize(stream);
//
//        {
//            std::lock_guard<std::mutex> frameLock(queueMutex);
//            frameQueue.push(frameBuffer);
//        }
//        queueCV.notify_one();
//
//        timer.update();
//        videoShowProgress(video, timer);
//        video.nextFrame();
//    }
//
//    isProcessing = false;
//    queueCV.notify_one();
//    writer.join();
//
//    // clean up
//    video.release();
//    cudaFree(d_img);
//    cudaStreamDestroy(stream);
//
//    mergeAudio(temp_video_name, temp_audio_name, outputPath);
//
//    fileUtils::deleteFile(temp_video_name);
//    fileUtils::deleteFile(temp_audio_name);
//}
//
//__host__ void videoPassColors(
//    const std::wstring& inputPath,
//    const std::wstring& outputPath,
//    const float* __restrict passThreshValues
//) {
//    // Generate temporary file names
//    std::wstring current_time = std::to_wstring(std::time(nullptr));
//
//    std::wstring video_root = fileUtils::splitextw(inputPath).first;
//    std::wstring output_ext = fileUtils::splitextw(outputPath).second;
//
//    std::wstring temp_video_name = video_root + L" " + current_time + output_ext;
//    std::wstring temp_audio_name = video_root + L" " + current_time + L".aac";
//
//    // Extract audio
//    extractAudio(inputPath, temp_audio_name);
//
//    Video video(inputPath, temp_video_name);
//    Timer timer;
//
//    unsigned char* d_img;
//    float* d_passThreshValues;
//
//    static constexpr size_t color_size = 3ULL * sizeof(float);
//
//    // Allocate device memory
//    checkCudaError(cudaMalloc(&d_img, video.getSize()), "Failed to allocate device memory for image");
//    checkCudaError(cudaMalloc(&d_passThreshValues, color_size), "Failed to allocate device memory for colors");
//
//    checkCudaError(cudaMemcpy(d_passThreshValues, passThreshValues, color_size, cudaMemcpyHostToDevice), "Failed to copy colors to device");
//
//    // Frame buffer pool (preallocated)
//    const int NUM_BUFFERS = nBuffers;
//    std::queue<cv::Mat> bufferPool;
//    for (int i = 0; i < NUM_BUFFERS; i++) {
//        cv::Mat frame(video.getHeight(), video.getWidth(), video.getType());
//        bufferPool.push(frame);
//    }
//
//    std::queue<cv::Mat> frameQueue;
//    std::mutex queueMutex;
//    std::condition_variable queueCV;
//    std::atomic<bool> isProcessing(true);
//
//    std::mutex bufferMutex;
//    std::condition_variable bufferCV;
//
//    // Writer thread function
//    auto writerThread = [&]() {
//        while (true) {
//            cv::Mat frame;
//            {
//                std::unique_lock<std::mutex> lock(queueMutex);
//                queueCV.wait(lock, [&]() { return !frameQueue.empty() || !isProcessing; });
//
//                if (!isProcessing && frameQueue.empty()) break;
//
//                frame = frameQueue.front();
//                frameQueue.pop();
//            }
//
//            video.write(frame);
//
//            // Recycle buffer
//            {
//                std::lock_guard<std::mutex> bufferLock(bufferMutex);
//                bufferPool.push(frame);
//                bufferCV.notify_one();
//            }
//        }
//        };
//
//    cudaStream_t stream;
//    checkCudaError(cudaStreamCreate(&stream), "Failed to create stream");
//
//    int blockSize = 1024;
//    int gridSize = (video.getNumPixels() + blockSize - 1) / blockSize;
//
//    std::thread writer(writerThread);
//
//    timer.start();
//    while (video.getSuccess()) {
//        std::unique_lock<std::mutex> bufferLock(bufferMutex);
//        bufferCV.wait(bufferLock, [&]() { return !bufferPool.empty(); });
//
//        cv::Mat frameBuffer = bufferPool.front();
//        bufferPool.pop();
//        bufferLock.unlock();
//
//        cudaMemcpyAsync(d_img, video.getData(), video.getSize(), cudaMemcpyHostToDevice, stream);
//
//        // fix intelisense
//        passColors_kernel<<<gridSize, blockSize, 0, stream>>>(d_img, video.getNumPixels(), d_passThreshValues);
//
//        cudaMemcpyAsync(frameBuffer.data, d_img, video.getSize(), cudaMemcpyDeviceToHost, stream);
//        cudaStreamSynchronize(stream);
//
//        {
//            std::lock_guard<std::mutex> frameLock(queueMutex);
//            frameQueue.push(frameBuffer);
//        }
//        queueCV.notify_one();
//
//        timer.update();
//        videoShowProgress(video, timer);
//        video.nextFrame();
//    }
//
//    isProcessing = false;
//    queueCV.notify_one();
//    writer.join();
//
//    // clean up
//    video.release();
//    cudaFree(d_img);
//    cudaStreamDestroy(stream);
//
//    mergeAudio(temp_video_name, temp_audio_name, outputPath);
//
//    fileUtils::deleteFile(temp_video_name);
//    fileUtils::deleteFile(temp_audio_name);
//}
