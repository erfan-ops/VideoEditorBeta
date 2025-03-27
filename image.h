//#pragma once
//
//#include <opencv2/opencv.hpp>
//
//
//class Image {
//public:
//    Image(Image&) = delete;
//    Image(const std::wstring& imagePath);
//    void save(const std::wstring& savePath);
//
//    // getters
//    size_t getSize() const noexcept;
//    size_t getNumPixels() const noexcept;
//    int getWidth() const noexcept;
//    int getHeight() const noexcept;
//    unsigned char* getData() const noexcept;
//
//    // properties
//    cv::Mat mat;
//private:
//    size_t size;
//    size_t nPixels;
//    int width;
//    int height;
//};
