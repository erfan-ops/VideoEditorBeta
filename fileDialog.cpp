#include "filedialog.h"

// Define the static member
const QString FileDialog::filters =
"Video Files (*.mp4 *.avi *.mkv *.mov);;"
"MP4 (*.mp4);;"
"AVI (*.avi);;"
"MKV (*.mkv);;"
"MOV (*.mov);;"
"Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.ppm *.pgm *.pbm *.hdr *.exr *.sr *.ras);;"
"JPEG (*.jpg *.jpeg);;"
"PNG (*.png);;"
"BMP (*.bmp);;"
"TIFF (*.tif *.tiff);;"
"WebP (*.webp);;"
"PPM/PGM/PBM (*.ppm *.pgm *.pbm);;"
"HDR (*.hdr);;"
"OpenEXR (*.exr);;"
"Sun Raster (*.sr *.ras);;"
"All Files (*.*)";

std::unordered_set<std::wstring> videoExtentions = { L".mp4", L".avi", L".mkv", L".mov" };
std::unordered_set<std::wstring> imageExtentions = {
    L".jpg", L".jpeg", L".png", L".bmp", L".tif", L".tiff", L".webp",
    L".ppm", L".pgm", L".pbm", L".hdr", L".exr", L".sr", L".ras"
};

// Helper functions (local to .cpp file)
namespace {
    QString toQString(const std::wstring& wstr) {
        return QString::fromStdWString(wstr);
    }

    std::wstring toStdWString(const QString& qstr) {
        return qstr.toStdWString();
    }
}

std::wstring FileDialog::OpenFileDialog(const std::wstring& selectedFilter) {
    QString selected = toQString(selectedFilter);
    QString file = QFileDialog::getOpenFileName(
        nullptr,                   // Parent widget
        "Open File",               // Dialog title
        "",                        // Default directory
        filters,                   // Use the class-static filters
        &selected                  // Preselect a filter
    );
    file.remove('"');
    return file.toStdWString();
}

std::wstring FileDialog::SaveFileDialog(const std::wstring& selectedFilter,
    const std::wstring& defaultName) {
    QString selected = toQString(selectedFilter);

    // Get native dialog without quotes
    QFileDialog dialog;
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setNameFilter(filters);
    dialog.selectNameFilter(selected);
    dialog.selectFile(toQString(defaultName));

    if (!dialog.exec()) {
        return L"";  // User cancelled
    }

    QString file = dialog.selectedFiles().first();
    file.remove('"');

    return QDir::toNativeSeparators(file).toStdWString();
}