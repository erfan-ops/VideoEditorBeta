#include "Video.h"
#include "utils.h"


static bool endsWith(cv::String str, cv::String suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}


Video::Video(const std::wstring& video_file_path, const std::wstring& video_output_path, int fourcc) {
    // Open video capture
    const cv::String video_file_path_utf8 = stringUtils::to_utf8(video_file_path);
    const cv::String video_output_path_utf8 = stringUtils::to_utf8(video_output_path);

    video_capture.open(video_file_path_utf8);
    if (!video_capture.isOpened()) {
        throw std::runtime_error("Error: Could not open video file: " + video_file_path_utf8);
    }

    // Get video properties
    FPS = video_capture.get(cv::CAP_PROP_FPS);
    if (FPS <= 0) {
        throw std::runtime_error("Error: Invalid FPS value in video file: " + video_file_path_utf8);
    }
    total_frames = static_cast<int>(video_capture.get(cv::CAP_PROP_FRAME_COUNT));
    total_video_duration = total_frames / FPS;
    frame_count = 0;

    width = static_cast<int>(video_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(video_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    imgSize = static_cast<unsigned long long>(width) * height * 3 * sizeof(unsigned char);

    // Set the fourcc codec
    if (fourcc != -1) {
        this->fourcc = fourcc;
    }
    else {
        if (endsWith(video_output_path_utf8, ".mp4")) {
            this->fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        }
        else if (endsWith(video_output_path_utf8, ".avi")) {
            this->fourcc = cv::VideoWriter::fourcc('I', '4', '2', '0');
        }
        else if (endsWith(video_output_path_utf8, ".mkv")) {
            this->fourcc = cv::VideoWriter::fourcc('h', 'e', 'v', '1');
        }
        else if (endsWith(video_output_path_utf8, ".mov")) {
            this->fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        }
        else {
            throw std::invalid_argument("Error: Unsupported video file extension: " + video_output_path_utf8);
        }
    }

    // Initialize VideoWriter
    video_write.open(video_output_path_utf8, this->fourcc, FPS, cv::Size(width, height));
    if (!video_write.isOpened()) {
        throw std::runtime_error("Error: Could not open video output file: " + video_output_path_utf8);
    }

    // Read the first frame
    nextFrame();
}

void Video::nextFrame() {
    success = video_capture.read(image);
    if (success) {
        frame_count++;
    }
    if (!image.isContinuous()) {
        image = image.clone();
    }
}

void Video::write(const cv::Mat& img) {
    video_write.write(img);
}

void Video::release() {
    video_capture.release();
    video_write.release();
}

int Video::get_frame_count() const { return frame_count; }
double Video::get_fps() const { return FPS; }
int Video::get_total_frames() const { return total_frames; }
double Video::get_total_video_duration() const { return total_video_duration; }
cv::Mat& Video::getImage() { return image; }
uchar* Video::getData() const { return image.data; }
int Video::getWidth() const { return width; }
int Video::getHeight() const { return height; }
size_t Video::getSize() const { return imgSize; }
bool Video::getSuccess() const { return success; }
