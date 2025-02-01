#include <iostream>
#include <Windows.h>

#include "videoEditor.cuh"
#include "utils.h"


int main() {
    SetConsoleOutputCP(CP_UTF8);  // Set output to UTF-8
    std::wcout.imbue(std::locale("en_US.UTF-8")); // Use the system's locale

    unsigned short pixelWidth, pixelHeight, lineWidth;
    unsigned char colorThresh, lineDarkness;
    unsigned char color1[3]{}, color2[3]{}, color3[3]{};

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

    std::cout << "Enter color1 (r g b, each 0-255): ";
    color1[2] = static_cast<unsigned char>(readInput("Red: ", 0, 255));
    color1[1] = static_cast<unsigned char>(readInput("Green: ", 0, 255));
    color1[0] = static_cast<unsigned char>(readInput("Blue: ", 0, 255));

    std::cout << "Enter color2 (r g b, each 0-255): ";
    color2[2] = static_cast<unsigned char>(readInput("Red: ", 0, 255));
    color2[1] = static_cast<unsigned char>(readInput("Green: ", 0, 255));
    color2[0] = static_cast<unsigned char>(readInput("Blue: ", 0, 255));

    std::cout << "Enter color3 (r g b, each 0-255): ";
    color3[2] = static_cast<unsigned char>(readInput("Red: ", 0, 255));
    color3[1] = static_cast<unsigned char>(readInput("Green: ", 0, 255));
    color3[0] = static_cast<unsigned char>(readInput("Blue: ", 0, 255));

    videoVintage8bit(
        inputPath, outputPath,
        pixelWidth, pixelHeight,
        color1, color2, color3,
        colorThresh,
        lineWidth, lineDarkness
    );

    return 0;
}
