#include "utils.h"
#include <Windows.h>
#include <string>


constinit static const wchar_t PROGRESS_STATES[8] = {L'▏', L'▎', L'▍', L'▌', L'▋', L'▊', L'▉', L'█'};
constinit static const int PROGRESS_STATES_LEN = sizeof(PROGRESS_STATES) / sizeof(wchar_t);
constinit static const int PROGRESS_STATES_LEN1 = PROGRESS_STATES_LEN - 1;
constinit static const int ESTIMATE_FROM_LAST_FRAMES = 30;

// Helper to convert seconds to MM:SS format
static std::wstring secondsToTimeW(float seconds) {
    int minutes = static_cast<int>(seconds) / 60;
    float remaining_seconds = seconds - minutes * 60;

    std::wstringstream wss;
    wss << minutes << L":" << std::fixed << std::setprecision(1)
        << std::setw(4) << std::setfill(L'0') << remaining_seconds;
    return wss.str();
}

// Display progress bar
void videoShowProgress(const Video& video, const Timer& timer, int batch_size) {
    // Get terminal width (default to 80 if not available)
    int progressBarLength = 80 - 50; // Adjust as needed

    float progressPct = static_cast<float>(video.get_frame_count()) / video.get_total_frames();
    int progress_in_mini = static_cast<int>(progressPct * (progressBarLength * PROGRESS_STATES_LEN + PROGRESS_STATES_LEN1));
    int full_bars = progress_in_mini / PROGRESS_STATES_LEN;

    // Build progress bar
    std::wstring progress_bar = std::wstring(full_bars, L'█') + PROGRESS_STATES[progress_in_mini - static_cast<std::vector<wchar_t, std::allocator<wchar_t>>::size_type>(full_bars) * PROGRESS_STATES_LEN];
    progress_bar += std::wstring(progressBarLength - full_bars, L' ');

    // Calculate estimate time
    float avg_time_per_frame = std::accumulate(timer.getPreviousTimes().begin(), timer.getPreviousTimes().end(), 0.0f) / timer.getPreviousTimes().size();
    float estimate_time = (video.get_total_frames() - video.get_frame_count()) * avg_time_per_frame / batch_size;

    // Print progress
    std::wcout << L"\r"
        << secondsToTimeW(timer.getTimeElapsed()) << L" elapsed |"
        << progress_bar << L"| "
        << video.get_frame_count() << L"/" << video.get_total_frames()
        << L" (" << std::fixed << std::setprecision(1) << progressPct * 100 << L"%) ["
        << secondsToTimeW(static_cast<float>(video.get_frame_count()) / video.get_fps()) << L"/"
        << secondsToTimeW(video.get_total_video_duration()) << L"] estimate time: "
        << secondsToTimeW(estimate_time) << L"   " << std::flush;
}


std::string wideStringToUtf8(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

// Function to execute a command with Unicode support
int execute_command(const std::wstring& command) {
    // Convert the command to a wide-character string
    std::wstring wide_command = L"cmd.exe /c " + command;

    // Initialize the STARTUPINFO and PROCESS_INFORMATION structures
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    // Create the process
    if (CreateProcess(
        nullptr,                  // No module name (use command line)
        &wide_command[0],         // Command line (wide-character)
        nullptr,                  // Process handle not inheritable
        nullptr,                  // Thread handle not inheritable
        FALSE,                    // Set handle inheritance to FALSE
        0,                        // No creation flags
        nullptr,                  // Use parent's environment block
        nullptr,                  // Use parent's starting directory
        &si,                      // Pointer to STARTUPINFO structure
        &pi                       // Pointer to PROCESS_INFORMATION structure
    )) {
        // Wait for the process to finish
        WaitForSingleObject(pi.hProcess, INFINITE);

        // Close process and thread handles
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        return 0; // Success
    }
    else {
        // Handle error
        std::cerr << "Failed to execute command. Error: " << GetLastError() << std::endl;
        return -1; // Failure
    }
}


std::pair<std::string, std::string> fileUtils::splitext(const std::string& path) {
    // Find the last '.' in the path
    size_t dotPos = path.find_last_of('.');
    size_t slashPos = path.find_last_of("/\\"); // Handle both '/' and '\\' for cross-platform paths

    // Check if the dot is part of the filename (not a directory name)
    if (dotPos != std::string::npos && (slashPos == std::string::npos || dotPos > slashPos)) {
        // Split the path into root and extension
        return { path.substr(0, dotPos), path.substr(dotPos) };
    }

    // If no valid extension is found, return the full path as the root and an empty string as the extension
    return { path, "" };
}

std::pair<std::wstring, std::wstring> fileUtils::splitextw(const std::wstring& path) {
    // Find the last '.' in the path
    size_t dotPos = path.find_last_of(L'.');
    size_t slashPos = path.find_last_of(L"/\\"); // Handle both '/' and '\\' for cross-platform paths

    // Check if the dot is part of the filename (not a directory name)
    if (dotPos != std::wstring::npos && (slashPos == std::wstring::npos || dotPos > slashPos)) {
        // Split the path into root and extension
        return { path.substr(0, dotPos), path.substr(dotPos) };
    }

    // If no valid extension is found, return the full path as the root and an empty string as the extension
    return { path, L"" };
}

std::string fileDialog::OpenFileDialog() {
    OPENFILENAMEA ofn;
    char fileName[MAX_PATH] = "";

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrTitle = "select the input video file";
    ofn.lpstrFilter = "Video Files\0*.mp4;*.avi;*.mkv;*.mov\0mp4\0*.mp4\0avi\0*.avi\0mkv\0*.mkv\0mov\0*.mov\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(fileName);
    }
    else {
        return ""; // User canceled the dialog
    }
}

std::wstring fileDialog::OpenFileDialogW() {
    OPENFILENAMEW ofnw;
    wchar_t fileName[MAX_PATH] = L"";

    ZeroMemory(&ofnw, sizeof(ofnw));
    ofnw.lStructSize = sizeof(ofnw);
    ofnw.hwndOwner = NULL;
    ofnw.lpstrTitle = L"select the input video file";
    ofnw.lpstrFilter = L"Video Files\0*.mp4;*.avi;*.mkv;*.mov\0mp4\0*.mp4\0avi\0*.avi\0mkv\0*.mkv\0mov\0*.mov\0";
    ofnw.lpstrFile = fileName;
    ofnw.nMaxFile = MAX_PATH;
    ofnw.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (GetOpenFileNameW(&ofnw)) {
        return std::wstring(fileName);
    }
    else {
        return L""; // User canceled the dialog
    }
}


std::string fileDialog::SaveFileDialog() {
    OPENFILENAMEA ofn;
    char fileName[MAX_PATH] = "output.mp4";

    // Initialize the OPENFILENAME structure
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL; // Handle to the owner window (NULL for no owner)
    ofn.lpstrTitle = "select the input video file";
    ofn.lpstrFilter = "Video Files\0*.mp4;*.avi;*.mkv;*.mov\0mp4\0*.mp4\0avi\0*.avi\0mkv\0*.mkv\0mov\0*.mov\0";
    ofn.lpstrFile = fileName; // Buffer to store the selected file path
    ofn.nMaxFile = MAX_PATH; // Size of the buffer
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST; // Flags for the dialog
    ofn.lpstrDefExt = "mp4";

    // Open the Save As dialog
    if (GetSaveFileNameA(&ofn)) {
        return std::string(fileName); // Return the selected file path
    }
    else {
        return ""; // User canceled the dialog
    }
}

std::wstring fileDialog::SaveFileDialogW() {
    OPENFILENAMEW ofnw;
    wchar_t fileName[MAX_PATH] = L"output.mp4";

    // Initialize the OPENFILENAME structure
    ZeroMemory(&ofnw, sizeof(ofnw));
    ofnw.lStructSize = sizeof(ofnw);
    ofnw.hwndOwner = NULL; // Handle to the owner window (NULL for no owner)
    ofnw.lpstrTitle = L"select the input video file";
    ofnw.lpstrFilter = L"Video Files\0*.mp4;*.avi;*.mkv;*.mov\0mp4\0*.mp4\0avi\0*.avi\0mkv\0*.mkv\0mov\0*.mov\0";
    ofnw.lpstrFile = fileName; // Buffer to store the selected file path
    ofnw.nMaxFile = MAX_PATH; // Size of the buffer
    ofnw.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST; // Flags for the dialog
    ofnw.lpstrDefExt = L"mp4";

    // Open the Save As dialog
    if (GetSaveFileNameW(&ofnw)) {
        return std::wstring(fileName); // Return the selected file path
    }
    else {
        return L""; // User canceled the dialog
    }
}


std::string stringUtils::to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), nullptr, 0, nullptr, nullptr);
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &str[0], size_needed, nullptr, nullptr);
    return str;
}

std::wstring stringUtils::string_to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();

    // Get the required size of the wide-character buffer
    int size_needed = MultiByteToWideChar(CP_ACP, 0, str.c_str(), (int)str.size(), nullptr, 0);

    // Allocate the wide-character buffer
    std::wstring wstr(size_needed, 0);

    // Perform the conversion
    MultiByteToWideChar(CP_ACP, 0, str.c_str(), (int)str.size(), &wstr[0], size_needed);

    return wstr;
}
