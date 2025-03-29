#include "mainWindow.h"
#include "ui_mainwindow.h"

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QMessageBox>
#include <QProgressBar>
#include <QThread>
#include <QPixmap>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>

#include <QDebug>

#include "filedialog.h"
#include "effects.h"
#include "utils.h"

#include "EffectButton.h"


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/icon.ico"));

    // Connect signals
    QObject::connect(ui->btnSelect, &QPushButton::clicked, [&]() {
        selectedFilePath = FileDialog::OpenFileDialog(L"All Files");
        if (!selectedFilePath.empty()) {
            ui->label->setText(QString::fromStdWString(L"Selected: " + selectedFilePath));
        }
        });

    replaceButtonWithEffectButton(ui->btnBlur, ":/samples/samples/blur.jpg");
    replaceButtonWithEffectButton(ui->btnOutlines, ":/samples/samples/outline.jpg");
    replaceButtonWithEffectButton(ui->btnPixelate, ":/samples/samples/pixelate.jpg");
    replaceButtonWithEffectButton(ui->btnCensor, ":/samples/samples/censor.jpg");
    replaceButtonWithEffectButton(ui->btnInverseColors, ":/samples/samples/inverseColors.jpg");
    replaceButtonWithEffectButton(ui->btnInverseContrast, ":/samples/samples/inverseContrast.jpg");
    replaceButtonWithEffectButton(ui->btnHueShift, ":/samples/samples/hueShift.jpg");
    replaceButtonWithEffectButton(ui->btnRadialBlur, ":/samples/samples/radialBlur.jpg");

    QObject::connect(ui->btnBlur, &QPushButton::clicked, this, [&]() {
        // Get effect parameters
        int blurRadius = ui->blurRadius->value();

        // Create the appropriate worker
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VBlurWorker(blurRadius);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IBlurWorker(blurRadius);
        }

        // Start processing
        processEffect(ui->btnBlur, worker);
        });

    QObject::connect(ui->btnOutlines, &QPushButton::clicked, this, [&]() {
        // Get effect parameters
        const int thicknessX = ui->ThicknessX->value();
        const int thicknessY = ui->ThicknessY->value();

        // Create the appropriate worker
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VOutlineWorker(thicknessX, thicknessY);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IOutlinesWorker(thicknessX, thicknessY);
        }

        // Start processing
        processEffect(ui->btnOutlines, worker);
        });

    QObject::connect(ui->btnPixelate, &QPushButton::clicked, this, [&]() {
        // Get effect parameters
        const int pixelWidth = ui->pixelWidth->value();
        const int pixelHeight = ui->pixelHeight->value();

        // Create the appropriate worker
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VPixelateWorker(pixelWidth, pixelHeight);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IPixelateWorker(pixelWidth, pixelHeight);
        }

        // Start processing
        processEffect(ui->btnPixelate, worker);
        });

    QObject::connect(ui->btnCensor, &QPushButton::clicked, this, [&]() {
        // Get effect parameters
        const int pixelWidth = ui->censorWidth->value();
        const int pixelHeight = ui->censorHeight->value();

        // Create the appropriate worker
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VCensorWorker(pixelWidth, pixelHeight);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new ICensorWorker(pixelWidth, pixelHeight);
        }

        // Start processing
        processEffect(ui->btnCensor, worker);
        });

    QObject::connect(ui->btnInverseColors, &QPushButton::clicked, this, [&]() {
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VInverseColorsWorker();
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IInverseColorsWorker();
        }

        processEffect(ui->btnInverseColors, worker);
        });

    QObject::connect(ui->btnHueShift, &QPushButton::clicked, this, [&]() {
        float shift = ui->HueShift->value();

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VHueShiftWorker(shift);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IHueShiftWorker(shift);
        }

        processEffect(ui->btnHueShift, worker);
        });

    QObject::connect(ui->btnRadialBlur, &QPushButton::clicked, this, [&]() {
        int blurRadius = ui->radialBlurRadius->value();
        float intensity = static_cast<float>(ui->Intensity->value());
        float centerX = static_cast<float>(ui->centerX->value());
        float centerY = static_cast<float>(ui->centerY->value());

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VRadialBlurWorker(blurRadius, intensity, centerX, centerY);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IRadialBlurWorker(blurRadius, intensity, centerX, centerY);
        }

        processEffect(ui->btnRadialBlur, worker);
        });
}

MainWindow::~MainWindow()
{
    delete ui;
}


void  MainWindow::processEffect(QPushButton* button, EffectBase* worker) {
    if (selectedFilePath.empty()) {
        QMessageBox::warning(this, "Error", "Please select a file first!");
        return;
    }

    QString fileType;

    if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
        saveFilePath = FileDialog::SaveFileDialog(L"Video Files", L"output.mp4");
        fileType = "Video";
    }
    else {
        saveFilePath = FileDialog::SaveFileDialog(L"Image Files", L"output.png");
        fileType = "Image";
    }
    if (saveFilePath.empty()) return;

    button->setEnabled(false);

    QThread* workerThread = new QThread();
    worker->setInputPath(selectedFilePath);
    worker->setOutputPath(saveFilePath);
    worker->moveToThread(workerThread);

    QObject::connect(workerThread, &QThread::started, worker, &EffectBase::process);
    QObject::connect(worker, &EffectBase::finished, workerThread, &QThread::quit);
    QObject::connect(worker, &EffectBase::errorOccurred, workerThread, &QThread::quit);

    QObject::connect(worker, &EffectBase::finished, this, [this, btn = QPointer<QPushButton>(button), fileType]() {
        btn->setEnabled(true);
        QMessageBox::information(this, "Success", fileType + " saved at \"" + QString::fromStdWString(saveFilePath) + '"');
        });

    QObject::connect(workerThread, &QThread::finished, worker, &EffectBase::deleteLater);
    QObject::connect(workerThread, &QThread::finished, workerThread, &QThread::deleteLater);

    workerThread->start();
}


void MainWindow::replaceButtonWithEffectButton(QPushButton*& button, const QString& imagePath)
{
    if (!button) return; // Safety check

    QWidget* parentWidget = button->parentWidget(); // Get the parent widget (important for QScrollArea)
    if (!parentWidget) return;

    // Create and set up the new EffectButton
    EffectButton* effectBtn = new EffectButton(imagePath, parentWidget);
    effectBtn->setCursor(Qt::PointingHandCursor);
    effectBtn->setStyleSheet("border: none; background: none;");

    // Set position relative to the parent widget
    effectBtn->setFixedSize(button->size());
    effectBtn->move(button->pos()); // Use move() instead of setGeometry()

    effectBtn->show();

    delete button;  // Remove the original button
    button = effectBtn; // Update pointer to the new button
}


void MainWindow::updateProgress(const Video& video, const Timer& timer) {
    ui->progressBar->setValue(video.get_frame_count() * 100 / video.get_total_frames());
    ui->elapsedTime->setText(QString::fromStdWString(secondsToTimeW(timer.getTimeElapsed())));

    float avg_time_per_frame = std::accumulate(timer.getPreviousTimes().begin(), timer.getPreviousTimes().end(), 0.0f) / timer.getPreviousTimes().size();
    float estimate_time = (video.get_total_frames() - video.get_frame_count()) * avg_time_per_frame;

    std::wstring details = std::to_wstring(
        video.get_frame_count()) +
        L"/" + std::to_wstring(video.get_total_frames()) +
        L" [" + secondsToTimeW(static_cast<float>(video.get_frame_count()) / video.get_fps()) +
        L"/" + secondsToTimeW(video.get_total_video_duration()) + L"] estimated time: " + secondsToTimeW(estimate_time);

    ui->progressDetails->setText(QString::fromStdWString(details));
}
