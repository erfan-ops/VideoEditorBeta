#pragma once

#include <QFileDialog>
#include <QWidget>

class FileDialog {
private:
    static const QString filters;
public:
    static std::wstring OpenFileDialog(const std::wstring& selectedFilter = L"");
    static std::wstring SaveFileDialog(const std::wstring& selectedFilter = L"", const std::wstring& defaultName = L"");
};
