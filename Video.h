#pragma once

#include <opencv2/opencv.hpp>


class Video {
public:
    Video(const std::wstring& video_file_path, const std::wstring& video_output_path, int fourcc = -1);
    void nextFrame();

    // Write the frame to the output video
    void write(const cv::Mat& img);

    // Release resources
    void release();

    // Getters for video properties
    int get_frame_count() const;
    double get_fps() const;
    int get_total_frames() const;
    double get_total_video_duration() const;
    cv::Mat& getImage();
    uchar* getData() const;
    int getWidth() const;
    int getHeight() const;
    size_t getSize() const;
    bool getSuccess() const;

private:
    cv::VideoCapture video_capture;
    cv::VideoWriter video_write;
    int fourcc;
    double FPS;
    int total_frames;
    double total_video_duration;
    int frame_count;
    int width, height;
    bool success;
    cv::Mat image;
    size_t imgSize;
};
