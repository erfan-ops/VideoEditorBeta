#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>

#include "videoEditor.cuh"
#include "utils.h"


int main(int argc, char** argv) {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0] << " <pixelWidth> <pixelHeight> <colorThresh> <lineWidth> <lineDarkness> <color1_r> <color1_g> <color1_b> <color2_r> <color2_g> <color2_b> <color3_r> <color3_g> <color3_b>" << std::endl;
        return 1;
    }

    unsigned short pixelWidth = static_cast<unsigned short>(std::stoi(argv[1]));
    unsigned short pixelHeight = static_cast<unsigned short>(std::stoi(argv[2]));
    unsigned char colorThresh = static_cast<unsigned char>(std::stoi(argv[3]));
    unsigned short lineWidth = static_cast<unsigned short>(std::stoi(argv[4]));
    unsigned char lineDarkness = static_cast<unsigned char>(std::stoi(argv[5]));

    unsigned char color1[3] = { static_cast<unsigned char>(std::stoi(argv[6])), static_cast<unsigned char>(std::stoi(argv[7])), static_cast<unsigned char>(std::stoi(argv[8])) };
    unsigned char color2[3] = { static_cast<unsigned char>(std::stoi(argv[9])), static_cast<unsigned char>(std::stoi(argv[10])), static_cast<unsigned char>(std::stoi(argv[11])) };
    unsigned char color3[3] = { static_cast<unsigned char>(std::stoi(argv[12])), static_cast<unsigned char>(std::stoi(argv[13])), static_cast<unsigned char>(std::stoi(argv[14])) };


    std::wstring inputPath = fileDialog::OpenFileDialogW();
    if (inputPath == L"")
        return 0;

    std::wstring outputPath = fileDialog::SaveFileDialogW();
    if (outputPath == L"")
        return 0;

	videoVintage8bit3(
        inputPath, outputPath,
        pixelWidth, pixelHeight,
        color1, color2, color3,
        colorThresh,
        lineWidth, lineDarkness
    );

    return 0;
}
