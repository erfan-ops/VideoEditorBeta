#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QMessageBox>
#include <QProgressBar>

#include "filedialog.h"
#include "videoEditor.cuh"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Create main window
    QWidget window;
    window.setWindowTitle("Video Processor with Settings");
    window.resize(400, 300);

    // Create UI elements
    QPushButton btnSelect("Select Video File");
    QPushButton btnOutlines("Generate Outlines");
    QLabel label("No file selected yet");

    // Number input widgets
    QSpinBox intInput;
    QLabel intLabel("Threshold:");

    // Configure number inputs
    intInput.setMinimum(1);
    intInput.setValue(1);               // Default value

    // Create layout
    QVBoxLayout layout(&window);
    layout.addWidget(&btnSelect);
    layout.addWidget(&btnOutlines);
    layout.addWidget(&label);

    // Add number inputs with labels
    layout.addWidget(&intLabel);
    layout.addWidget(&intInput);

    QProgressBar progressBar;
    progressBar.setRange(0, 100);  // 0-100%
    progressBar.setValue(0);       // Start at 0%
    progressBar.setTextVisible(true);  // Show percentage text
    layout.addWidget(&progressBar);

    // Variables to store values
    std::wstring selectedFilePath;
    int Thresh{ 0 };

    // Connect signals
    QObject::connect(&btnSelect, &QPushButton::clicked, [&]() {
        selectedFilePath = FileDialog::OpenFileDialog(L"Video Files");
        if (!selectedFilePath.empty()) {
            label.setText(QString::fromStdWString(L"Selected: " + selectedFilePath));
        }
        });

    QObject::connect(&btnOutlines, &QPushButton::clicked, [&]() {
        if (selectedFilePath.empty()) {
            QMessageBox::warning(&window, "Error", "Please select a video file first!");
            return;
        }

        // Get current values from spin boxes
        Thresh = intInput.value();

        qDebug() << "Processing with Thresh:" << Thresh;

        std::wstring savePath = FileDialog::SaveFileDialog(L"Video Files", L"video.mp4");
        if (!savePath.empty()) {
            videoOutlines(selectedFilePath, savePath, Thresh, Thresh);
            QMessageBox::information(&window, "Success",
                QString("Processed with Thresh %1")
                .arg(Thresh));
        }
        });

    window.show();
    return app.exec();
}