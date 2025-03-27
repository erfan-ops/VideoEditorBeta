#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QMessageBox>
#include <QProgressBar>
#include <QThread>

#include <QDebug>

#include "filedialog.h"
#include "videoOutlines.h"

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
    QSpinBox shiftXInput;
    QSpinBox shiftYInput;
    QLabel shiftXLabel("Shift X:");
    QLabel shiftYLabel("Shift Y:");

    // Configure number inputs
    shiftXInput.setMinimum(1);
    shiftXInput.setMaximum(100);
    shiftXInput.setValue(1);

    shiftYInput.setMinimum(1);
    shiftYInput.setMaximum(100);
    shiftYInput.setValue(1);

    // Create layout
    QVBoxLayout layout(&window);
    layout.addWidget(&btnSelect);
    layout.addWidget(&btnOutlines);
    layout.addWidget(&label);

    // Add number inputs with labels
    layout.addWidget(&shiftXLabel);
    layout.addWidget(&shiftXInput);
    layout.addWidget(&shiftYLabel);
    layout.addWidget(&shiftYInput);

    QProgressBar progressBar;
    progressBar.setRange(0, 100);
    progressBar.setValue(0);
    progressBar.setTextVisible(true);
    layout.addWidget(&progressBar);

    // Variables to store values
    std::wstring selectedFilePath;

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

        std::wstring savePath = FileDialog::SaveFileDialog(L"Video Files", L"video_outlines.mp4");
        if (savePath.empty()) return;

        // Disable button during processing
        btnOutlines.setEnabled(false);

        // Get current values
        int shiftX = shiftXInput.value();
        int shiftY = shiftYInput.value();

        // Create worker and thread
        QThread* workerThread = new QThread();
        OutlineWorker* worker = new OutlineWorker(shiftX, shiftY);

        // Set file paths
        worker->setInputPath(selectedFilePath);
        worker->setOutputPath(savePath);
        worker->moveToThread(workerThread);

        // Connect signals
        QObject::connect(workerThread, &QThread::started, worker, &OutlineWorker::process);
        QObject::connect(worker, &OutlineWorker::progressChanged, &progressBar, &QProgressBar::setValue, Qt::QueuedConnection);


        QObject::connect(worker, &OutlineWorker::finished, workerThread, &QThread::quit);
        QObject::connect(worker, &OutlineWorker::errorOccurred, workerThread, &QThread::quit);

        // And update the UI separately if needed:
        QObject::connect(worker, &OutlineWorker::finished, &window, [&]() {
            btnOutlines.setEnabled(true);
            });
        QObject::connect(worker, &OutlineWorker::errorOccurred, &window, [&](const QString& error) {
            btnOutlines.setEnabled(true);
            });


        // Make sure workerThread is deleted properly
        QObject::connect(workerThread, &QThread::finished, worker, &OutlineWorker::deleteLater);
        QObject::connect(workerThread, &QThread::finished, workerThread, &QThread::deleteLater);


        // Start processing
        workerThread->start();
        });

    window.show();
    return app.exec();
}