#include <iostream>
#include <Windows.h>

#include "videoEditor.cuh"
#include "utils.h"


int main() {
    SetConsoleOutputCP(CP_UTF8);  // Set output to UTF-8
    std::wcout.imbue(std::locale("en_US.UTF-8")); // Use the system's locale

    unsigned short pixelWidth, pixelHeight, lineWidth;
    unsigned char colorThresh, lineDarkness;
    int nColors;

    // Function to safely read an integer within a specified range
    auto readInput = [](const std::string& prompt, int minVal, int maxVal) -> int {
        int value;
        while (true) {
            std::cout << prompt;
            std::cin >> value;
            if (std::cin.fail() || value < minVal || value > maxVal) {
                std::cin.clear(); // Clear the error flag
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                std::cerr << "Invalid input. Please enter a number between " << minVal << " and " << maxVal << "." << std::endl;
            }
            else {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard any extra input
                return value;
            }
        }
        };

    // File dialogs for input and output paths
    std::wstring inputPath = fileDialog::OpenFileDialogW();
    if (inputPath == L"") {
        std::cerr << "No input file selected." << std::endl;
        return 0;
    }

    std::wstring outputPath = fileDialog::SaveFileDialogW();
    if (outputPath == L"") {
        std::cerr << "No output file selected." << std::endl;
        return 0;
    }

    // Prompt the user for each parameter
    pixelWidth = static_cast<unsigned short>(readInput("Enter pixel width: ", 0, 65535));
    pixelHeight = static_cast<unsigned short>(readInput("Enter pixel height: ", 0, 65535));
    colorThresh = static_cast<unsigned char>(readInput("Enter color threshold: ", 0, 255));
    lineWidth = static_cast<unsigned short>(readInput("Enter line width: ", 0, 65535));
    lineDarkness = static_cast<unsigned char>(readInput("Enter line darkness: ", 0, 255));
    nColors = static_cast<int>(readInput("Enter number of colors: ", 0, 100));

    unsigned char* colors_BGR = new unsigned char[nColors * 3];


    for (int i = 0; i < nColors; ++i) {
        std::cout << "Enter color" << i+1 << " (r g b, each 0 - 255) : ";
        int idxFactor = i * 3;
        colors_BGR[2+idxFactor] = static_cast<unsigned char>(readInput("Red: ", 0, 255));
        colors_BGR[1+idxFactor] = static_cast<unsigned char>(readInput("Green: ", 0, 255));
        colors_BGR[0+idxFactor] = static_cast<unsigned char>(readInput("Blue: ", 0, 255));
    }

    videoVintage8bit(
        inputPath, outputPath,
        pixelWidth, pixelHeight,
        colors_BGR, nColors,
        colorThresh,
        lineWidth, lineDarkness
    );

    delete[] colors_BGR;

    return 0;
}
