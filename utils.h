#pragma once

#include "Video.h"
#include "Timer.h"

#include <string>
#include <QString>


void videoShowProgress(const Video& video, const Timer& timer, int batch_size = 1);

int execute_command(const std::wstring& command);
int Qexecute_command(const QString& program, const QStringList& arguments);

std::string wideStringToUtf8(const std::wstring& wstr);

namespace fileUtils {
    std::pair<std::string, std::string> splitext(const std::string& path);
    std::pair<std::wstring, std::wstring> splitextw(const std::wstring& path);
    void deleteFile(const std::wstring& path);
}

namespace stringUtils {
    std::string to_utf8(const std::wstring& wstr);
    std::wstring string_to_wstring(const std::string& str);
}
