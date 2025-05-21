#include "utils.h"

#include <Windows.h>
#include <numeric>
#include <codecvt>
#include <filesystem>

#include <QProcess>


constinit static const wchar_t PROGRESS_STATES[8] = {L'▏', L'▎', L'▍', L'▌', L'▋', L'▊', L'▉', L'█'};
constinit static const int PROGRESS_STATES_LEN = sizeof(PROGRESS_STATES) / sizeof(wchar_t);
constinit static const int PROGRESS_STATES_LEN1 = PROGRESS_STATES_LEN - 1;
constinit static const int ESTIMATE_FROM_LAST_FRAMES = 30;


// Helper to convert seconds to MM:SS format
std::wstring secondsToTimeW(float seconds) {
    int minutes = static_cast<int>(seconds) / 60;
    float remaining_seconds = seconds - minutes * 60;

    std::wstringstream wss;
    wss << minutes << L":" << std::fixed << std::setprecision(1)
        << std::setw(4) << std::setfill(L'0') << remaining_seconds;
    return wss.str();
}


std::string wideStringToUtf8(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

int execute_command(const std::wstring& command) {
    QProcess process;
    process.setProcessChannelMode(QProcess::MergedChannels); // Combine stdout/stderr

    process.start(QString::fromStdWString(command));

    if (!process.waitForStarted()) {
        return -1;
    }

    if (!process.waitForFinished()) {
        return -2;
    }

    return process.exitCode();
}

int Qexecute_command(const QString& program, const QStringList& arguments) {
    QProcess process;
    process.setProcessChannelMode(QProcess::MergedChannels);

    process.start(program, arguments);

    if (!process.waitForStarted()) {
        return -1;
    }

    if (!process.waitForFinished()) {
        return -2;
    }
    return process.exitCode();
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

void fileUtils::deleteFile(const std::wstring& path) {
    std::filesystem::remove(path);
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


void videoUtils::checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void videoUtils::extractAudio(const std::wstring& inputVideo, const std::wstring& outputAudio) {
    QProcess process;
    process.start("ffmpeg", {
        "-loglevel", "quiet",
        "-i", QString::fromStdWString(inputVideo),
        "-vn", "-acodec", "copy",
        QString::fromStdWString(outputAudio)
        });
    process.waitForFinished();
}

void videoUtils::mergeAudio(const std::wstring& inputVideo, const std::wstring& inputAudio, const std::wstring& outputVideo) {
    QProcess process;
    process.start(
        "ffmpeg", {
            "-loglevel", "quiet",
            "-i", QString::fromStdWString(inputVideo),
            "-i", QString::fromStdWString(inputAudio),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            QString::fromStdWString(outputVideo),
            "-y"
        });
    process.waitForFinished();
}

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

cl_context openclUtils::createContext(cl_device_id& device) {
    cl_uint numPlatforms;
    cl_platform_id platform = nullptr;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "Failed to find any OpenCL platforms.\n";
        return nullptr;
    }

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cerr << "No OpenCL GPU devices found.\n";
        return nullptr;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context.\n";
        return nullptr;
    }

    return context;
}

cl_command_queue openclUtils::createCommandQueue(cl_context context, cl_device_id device) {
    cl_int err;
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device, 0, &err
    );
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL command queue.\n";
        return nullptr;
    }
    return queue;
}

cl_kernel openclUtils::createKernelFromSource(cl_context context, cl_device_id device, const char* source, const char* kernelName) {
    cl_int err;
    size_t lengths[] = { strlen(source) };
    const char* sources[] = { source };

    cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program from source.\n";
        return nullptr;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Show build log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << "\n";
        return nullptr;
    }

    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL kernel.\n";
        return nullptr;
    }

    return kernel;
}
