#include <iostream>
#include <Windows.h>

#include "videoEditor.cuh"
#include "imageEditor.cuh"
#include "utils.h"


int main() {
    SetConsoleOutputCP(CP_UTF8);  // Set output to UTF-8
    std::wcout.imbue(std::locale("en_US.UTF-8")); // Use the system's locale

    // File dialogs for input and output paths
    std::wstring inputPath = fileDialog::OpenFileDialogW(L"Image Files");
    if (inputPath == L"") {
        std::cerr << "No input file selected." << std::endl;
        return 0;
    }

    std::wstring outputPath = fileDialog::SaveFileDialogW(L"Image Files");
    if (outputPath == L"") {
        std::cerr << "No output file selected." << std::endl;
        return 0;
    }

    videoTrueOutlines(inputPath, outputPath, 3);

    return 0;
}
