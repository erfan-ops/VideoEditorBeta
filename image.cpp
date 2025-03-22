#include "image.h"
#include "utils.h"


Image::Image(const std::wstring& imagePath) {
    mat = cv::imread(wideStringToUtf8(imagePath), cv::IMREAD_COLOR);

    if (mat.empty()) {
        throw std::runtime_error("Error: Could not open or find the image!");
    }

    width = mat.cols;
    height = mat.rows;

    nPixels = static_cast<size_t>(width) * height;
    size = nPixels * 3;
}

void Image::save(const std::wstring& savePath) {
    if (cv::imwrite(wideStringToUtf8(savePath), mat)) {
        std::cout << "Image saved successfully as 'output.jpg'" << std::endl;
    }
    else {
        std::cerr << "Error: Could not save the image!" << std::endl;
    }
}

size_t Image::getSize() const noexcept { return size; }
size_t Image::getNumPixels() const noexcept { return nPixels; }
int Image::getWidth() const noexcept { return width; }
int Image::getHeight() const noexcept { return height; }
unsigned char* Image::getData() const noexcept { return mat.data; }
