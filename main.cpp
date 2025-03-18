#include <iostream>
#include <Windows.h>

#include "videoEditor.cuh"
#include "utils.h"


int main() {
    SetConsoleOutputCP(CP_UTF8);  // Set output to UTF-8
    std::wcout.imbue(std::locale("en_US.UTF-8")); // Use the system's locale

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
    
    unsigned char colors_BGR[] = {
       55, 4, 45,
        72, 127, 182,
        162, 230, 255,
    };

    float passColors[] = { 1.0f, 0.5f, 0.0f };

    //videoVintage8bit(inputPath, outputPath, 1, 1, colors_BGR, 3, 64, 8, 0);
    videoPassColors(inputPath, outputPath, passColors);

    return 0;
}
