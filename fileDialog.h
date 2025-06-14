#pragma once

#include <QFileDialog>
#include <QWidget>

#include <unordered_set>

extern std::unordered_set<std::wstring> videoExtentions;
extern std::unordered_set<std::wstring> imageExtentions;

class FileDialog {
private:
    static const QString filters;
public:
    static std::wstring OpenFileDialog(const std::wstring& selectedFilter = L"");
    static std::wstring SaveFileDialog(const std::wstring& selectedFilter = L"", const std::wstring& defaultName = L"");
};
