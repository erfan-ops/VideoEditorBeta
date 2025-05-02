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
#include <QColorDialog>
#include <QImageReader>
#include <QScrollbar>

#include "filedialog.h"
#include "effects.h"
#include "launchers.h"
#include "utils.h"

#include "EffectButton.h"


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/icon.ico"));

    cudaStreamCreate(&this->streamHueShift);
    cudaStreamCreate(&this->streamFilter);
    cudaStreamCreate(&this->streamBinary);
    cudaStreamCreate(&this->streamInverseColors);
    cudaStreamCreate(&this->streamInverseContrast);
    cudaStreamCreate(&this->streamMonoChrome);
    cudaStreamCreate(&this->streamBlur);
    cudaStreamCreate(&this->streamOutLine);
    cudaStreamCreate(&this->streamTrueOutLine);
    cudaStreamCreate(&this->streamPosterize);
    cudaStreamCreate(&this->streamRadialBlur);
    cudaStreamCreate(&this->streamCensor);
    cudaStreamCreate(&this->streamPixelate);
    cudaStreamCreate(&this->streamVintage8bit);
    cudaStreamCreate(&this->streamChangePalette);
    cudaStreamCreate(&this->streamMonoMask);

    // Connect signals
    QObject::connect(ui->btnSelect, &QPushButton::clicked, [&]() {
        selectedFilePath = FileDialog::OpenFileDialog(L"All Files");
        if (!selectedFilePath.empty()) {
            ui->label->setText(QString::fromStdWString(L"Selected: " + selectedFilePath));

            if (!(videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()))
                this->newThumbnails();
        }
        });

    replaceButtonWithEffectButton(ui->btnBlur, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnOutlines, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnPixelate, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnCensor, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnInverseColors, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnInverseContrast, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnHueShift, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnRadialBlur, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnMonoChrome, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnBinary, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnPosterize, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnTrueOutlines, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnMagicEye, ":/samples/samples/noise.jpg");
    replaceButtonWithEffectButton(ui->btnVintage8bit, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnFilter, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnChangePalette, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnMonoMask, ":/samples/samples/sample.jpg");

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnRadialBlur);
    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    ui->centerX->setMaximum(originalPixmap.width());
    ui->centerY->setMaximum(originalPixmap.height());

    ui->centerX->setValue(0.5f * originalPixmap.width());
    ui->centerY->setValue(0.5f * originalPixmap.height());

    this->updateThumbnails();

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
    QObject::connect(ui->blurRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateBlurThumbnail);

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
    QObject::connect(ui->ThicknessX, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateOutLineThumbnail);
    QObject::connect(ui->ThicknessY, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateOutLineThumbnail);

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
    QObject::connect(ui->pixelWidth, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updatePixelateThumbnail);
    QObject::connect(ui->pixelHeight, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updatePixelateThumbnail);

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
    QObject::connect(ui->censorWidth, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateCensorThumbnail);
    QObject::connect(ui->censorHeight, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateCensorThumbnail);

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

    QObject::connect(ui->btnInverseContrast, &QPushButton::clicked, this, [&]() {
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VInverseContrastWorker();
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IInverseContrastWorker();
        }

        processEffect(ui->btnInverseContrast, worker);
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
    QObject::connect(ui->HueShift, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::updateHueShiftThumbnail);

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
    QObject::connect(ui->radialBlurRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateRadialBlurThumbnail);
    QObject::connect(ui->Intensity, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::updateRadialBlurThumbnail);
    QObject::connect(ui->centerX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::updateRadialBlurThumbnail);
    QObject::connect(ui->centerY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::updateRadialBlurThumbnail);

    QObject::connect(ui->btnMonoChrome, &QPushButton::clicked, this, [&]() {
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VMonoChromeWorker();
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IMonoChromeWorker();
        }

        processEffect(ui->btnMonoChrome, worker);
        });

    QObject::connect(ui->btnBinary, &QPushButton::clicked, this, [&]() {
        float middle = ui->middle->value();

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VBlackAndWhiteWorker(middle);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IBlackAndWhiteWorker(middle);
        }

        processEffect(ui->btnBinary, worker);
        });
    QObject::connect(ui->middle, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::updateBinaryThumbnail);

    QObject::connect(ui->btnPosterize, &QPushButton::clicked, this, [&]() {
        int thresh = ui->posterizeThresh->value();

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VPosterizeWorker(thresh);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IPosterizeWorker(thresh);
        }

        processEffect(ui->btnPosterize, worker);
        });
    QObject::connect(ui->posterizeThresh, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updatePosterizeThumbnail);

    QObject::connect(ui->btnTrueOutlines, &QPushButton::clicked, this, [&]() {
        int thresh = ui->trueOutlinesThresh->value();

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VTrueOutlinesWorker(thresh);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new ITrueOutlinesWorker(thresh);
        }

        processEffect(ui->btnTrueOutlines, worker);
        });
    QObject::connect(ui->trueOutlinesThresh, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateTrueOutLineThumbnail);

    QObject::connect(ui->btnMagicEye, &QPushButton::clicked, this, [&]() {
        float middle = ui->magicEyeMiddle->value();

        if (!(videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end())) {
            QMessageBox::warning(this, "Video Only Effect", "This effect is for videos only and cannot be applied to images!");
            return;
        }

        VMagicEyeWorker* worker = new VMagicEyeWorker(middle);
        QObject::connect(worker, &VMagicEyeWorker::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        processEffect(ui->btnMagicEye, worker);
        });

    QObject::connect(ui->btnVintage8bit, &QPushButton::clicked, this, [&]() {
        int pixelWidth = ui->vintagePixelWidth->value();
        int pixelHeight = ui->vintagePixelHeight->value();
        int thresh = ui->vintagePosterizeThresh->value();

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VVintage8bitWorker(pixelWidth, pixelHeight, thresh);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IVintage8bitWorker(pixelWidth, pixelHeight, thresh);
        }

        processEffect(ui->btnVintage8bit, worker);
        });
    QObject::connect(ui->vintagePixelWidth, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateVintage8bitThumbnail);
    QObject::connect(ui->vintagePixelHeight, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateVintage8bitThumbnail);
    QObject::connect(ui->vintagePosterizeThresh, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::updateVintage8bitThumbnail);

    QObject::connect(ui->filterSelectColor, &QPushButton::clicked, this, [&]() {
        QColor color = QColorDialog::getColor(filterColor, this, "Select a Color");
        if (color.isValid()) {
            filterColor = color;
            QColor hoverColor = color.darker(120); // 120 = 20% darker
            QString style = QString(
                "QPushButton { background-color: %1; border: none; border-radius: 6px; }"
                "QPushButton:hover { background-color: %2; }"
            ).arg(color.name(), hoverColor.name());

            ui->filterColorDisplay->setStyleSheet(style);

            this->updateFilterThumbnail();
        }
        });
    QObject::connect(ui->filterColorDisplay, &QPushButton::clicked, this, [&]() {
        QColor color = QColorDialog::getColor(filterColor, this, "Select a Color");
        if (color.isValid()) {
            filterColor = color;
            QColor hoverColor = color.darker(120); // 120 = 20% darker
            QString style = QString(
                "QPushButton { background-color: %1; border: none; border-radius: 6px; }"
                "QPushButton:hover { background-color: %2; }"
            ).arg(color.name(), hoverColor.name());

            ui->filterColorDisplay->setStyleSheet(style);

            this->updateFilterThumbnail();
        }
        });
    QObject::connect(ui->btnFilter, &QPushButton::clicked, this, [&]() {
        float passThreshValues[] = { filterColor.blueF(), filterColor.greenF(), filterColor.redF() };

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VLensFilterWorker(passThreshValues);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new ILensFilterWorker(passThreshValues);
        }

        processEffect(ui->btnFilter, worker);
        });

    if (!ui->changePaletteScrollAreaWidgetContents->layout()) {
        QVBoxLayout* layout = new QVBoxLayout(ui->changePaletteScrollAreaWidgetContents);
        layout->setSizeConstraint(QLayout::SetMinAndMaxSize);
        layout->setAlignment(Qt::AlignTop);
        ui->changePaletteScrollAreaWidgetContents->setLayout(layout);
    }

    QObject::connect(ui->changePaletteAddColorBtn, &QPushButton::clicked, this, [&]() {
        QColor color = QColorDialog::getColor(Qt::white, this, "Select a Color");
        if (color.isValid()) {
            this->changePaletteColorsVector.push_back(color.blue());
            this->changePaletteColorsVector.push_back(color.green());
            this->changePaletteColorsVector.push_back(color.red());

            // Create a styled label
            QLabel* colorLabel = new QLabel();
            colorLabel->setFixedSize(162, 20);
            colorLabel->setStyleSheet(QString(
                "background-color: %1; "
                "border-radius: 4px;"
            ).arg(color.name()));

            // Add to the layout of the scroll area's contents widget
            QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->changePaletteScrollAreaWidgetContents->layout());
            if (layout) {
                layout->addWidget(colorLabel);

                // Auto-scroll to bottom
                ui->changePaletteScrollArea->verticalScrollBar()->setValue(
                    ui->changePaletteScrollArea->verticalScrollBar()->maximum()
                );
            }

            this->updateChangePaletteThumbnail();
        }
        });
    QObject::connect(ui->btnChangePalette, &QPushButton::clicked, this, [&]() {
        unsigned char* colorsBGR = this->changePaletteColorsVector.data();
        int numColors = this->changePaletteColorsVector.size() / 3;

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VChangePaletteWorker(colorsBGR, numColors);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IChangePaletteWorker(colorsBGR, numColors);
        }

        processEffect(ui->btnChangePalette, worker);
        });

    if (!ui->monoMaskScrollAreaWidgetContents->layout()) {
        QVBoxLayout* layout = new QVBoxLayout(ui->monoMaskScrollAreaWidgetContents);
        layout->setSizeConstraint(QLayout::SetMinAndMaxSize);
        layout->setAlignment(Qt::AlignTop);
        ui->monoMaskScrollAreaWidgetContents->setLayout(layout);
    }

    QObject::connect(ui->monoMaskAddColorBtn, &QPushButton::clicked, this, [&]() {
        QColor color = QColorDialog::getColor(Qt::white, this, "Select a Color");
        if (color.isValid()) {
            this->monoMaskColorsVector.push_back(color.blue());
            this->monoMaskColorsVector.push_back(color.green());
            this->monoMaskColorsVector.push_back(color.red());

            // Create a styled label
            QLabel* colorLabel = new QLabel();
            colorLabel->setFixedSize(162, 20);
            colorLabel->setStyleSheet(QString(
                "background-color: %1; "
                "border-radius: 4px;"
            ).arg(color.name()));

            // Add to the layout of the scroll area's contents widget
            QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->monoMaskScrollAreaWidgetContents->layout());
            if (layout) {
                layout->addWidget(colorLabel);

                // Auto-scroll to bottom
                ui->monoMaskScrollArea->verticalScrollBar()->setValue(
                    ui->monoMaskScrollArea->verticalScrollBar()->maximum()
                );
            }

            this->updateMonoMaskThumbnail();
        }
        });
    QObject::connect(ui->btnMonoMask, &QPushButton::clicked, this, [&]() {
        unsigned char* colorsBGR = this->monoMaskColorsVector.data();
        int numColors = this->monoMaskColorsVector.size() / 3;

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VMonoMaskWorker(colorsBGR, numColors);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IMonoMaskWorker(colorsBGR, numColors);
        }

        processEffect(ui->btnMonoMask, worker);
        });
}

MainWindow::~MainWindow()
{
    cudaStreamDestroy(this->streamHueShift);
    cudaStreamDestroy(this->streamFilter);
    cudaStreamDestroy(this->streamBinary);
    cudaStreamDestroy(this->streamInverseColors);
    cudaStreamDestroy(this->streamInverseContrast);
    cudaStreamDestroy(this->streamMonoChrome);
    cudaStreamDestroy(this->streamBlur);
    cudaStreamDestroy(this->streamOutLine);
    cudaStreamDestroy(this->streamTrueOutLine);
    cudaStreamDestroy(this->streamPosterize);
    cudaStreamDestroy(this->streamRadialBlur);
    cudaStreamDestroy(this->streamCensor);
    cudaStreamDestroy(this->streamCensor);
    cudaStreamDestroy(this->streamPixelate);
    cudaStreamDestroy(this->streamVintage8bit);
    cudaStreamDestroy(this->streamChangePalette);
    cudaStreamDestroy(this->streamMonoMask);

    delete ui;
}


void MainWindow::processEffect(QPushButton* button, EffectBase* worker) {
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
    ui->progressBar->setValue(video.get_frame_count() * 1131 / video.get_total_frames());
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


void MainWindow::updateHueShiftThumbnail() {
    float rotationFactor = ui->HueShift->value() * 2.0f;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnHueShift);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;

    cudaMalloc(&d_img, size);

    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (nPixels + blockSize - 1) / blockSize;

    hueShiftRGBA(gridSize, blockSize, this->streamHueShift, d_img, nPixels, rotationFactor);
    cudaStreamSynchronize(this->streamHueShift);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updateFilterThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnFilter);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    unsigned char* bgrCopy = new unsigned char[size];
    memcpy(bgrCopy, image.constBits(), size);

    float passThreshValues[] = { filterColor.redF(), filterColor.greenF(), filterColor.blueF(), 1.0f };

    unsigned char* d_img;
    float* d_passThreshValues;

    cudaMalloc(&d_img, size);
    cudaMalloc(&d_passThreshValues, 4 * sizeof(float));

    cudaMemcpy(d_img, bgrCopy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_passThreshValues, passThreshValues, 4 * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;

    lensFilterRGBA(gridSize, blockSize, this->streamFilter, d_img, size, d_passThreshValues);
    cudaStreamSynchronize(this->streamFilter);

    cudaMemcpy(bgrCopy, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(bgrCopy, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] bgrCopy;
    cudaFree(d_img);
    cudaFree(d_passThreshValues);
}

void MainWindow::updateBinaryThumbnail() {
    float middle = ui->middle->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnBinary);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    unsigned char* bgrCopy = new unsigned char[size];
    memcpy(bgrCopy, image.constBits(), size);

    unsigned char* d_img;

    cudaMalloc(&d_img, size);

    cudaMemcpy(d_img, bgrCopy, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;

    blackAndWhiteRGBA(gridSize, blockSize, this->streamBinary, d_img, nPixels, middle);
    cudaStreamSynchronize(this->streamBinary);

    cudaMemcpy(bgrCopy, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(bgrCopy, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] bgrCopy;
    cudaFree(d_img);
}

void MainWindow::updateInverseColorsThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnInverseColors);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;

    inverseColorsRGBA(gridSize, blockSize, this->streamInverseColors, d_img, size);
    cudaStreamSynchronize(this->streamInverseColors);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updateInverseContrastThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnInverseContrast);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (nPixels + blockSize - 1) / blockSize;

    inverseContrastRGBA(gridSize, blockSize, this->streamInverseContrast, d_img, nPixels);
    cudaStreamSynchronize(this->streamInverseContrast);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updateMonoChromeThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnMonoChrome);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (nPixels + blockSize - 1) / blockSize;

    monoChromeRGBA(gridSize, blockSize, this->streamMonoChrome, d_img, nPixels);
    cudaStreamSynchronize(this->streamMonoChrome);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updateBlurThumbnail() {
    int radius = ui->blurRadius->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnBlur);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    unsigned char* d_img_copy;
    cudaMalloc(&d_img, size);
    cudaMalloc(&d_img_copy, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_copy, d_img, size, cudaMemcpyDeviceToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((image.width() + blockDim.x - 1) / blockDim.x, (image.height() + blockDim.y - 1) / blockDim.y);

    blurRGBA(gridDim, blockDim, this->streamBlur, d_img, d_img_copy, image.width(), image.height(), radius);
    cudaStreamSynchronize(this->streamBlur);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
    cudaFree(d_img_copy);
}

void MainWindow::updateOutLineThumbnail() {
    const int thicknessX = ui->ThicknessX->value();
    const int thicknessY = ui->ThicknessY->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnOutlines);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    unsigned char* d_img_copy;
    cudaMalloc(&d_img, size);
    cudaMalloc(&d_img_copy, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_copy, d_img, size, cudaMemcpyDeviceToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((image.width() + blockDim.x - 1) / blockDim.x, (image.height() + blockDim.y - 1) / blockDim.y);

    outlinesRGBA(gridDim, blockDim, this->streamOutLine, d_img, d_img_copy, image.width(), image.height(), thicknessX, thicknessY);
    cudaStreamSynchronize(this->streamOutLine);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
    cudaFree(d_img_copy);
}

void MainWindow::updateTrueOutLineThumbnail() {
    const int thresh = ui->trueOutlinesThresh->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnTrueOutlines);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    unsigned char* d_img_copy;
    cudaMalloc(&d_img, size);
    cudaMalloc(&d_img_copy, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_copy, d_img, size, cudaMemcpyDeviceToDevice);

    int blockSize = 1024;
    int gridSize = (nPixels + blockSize - 1) / blockSize;

    dim3 blockDim(32, 32);
    dim3 gridDim((image.width() + blockDim.x - 1) / blockDim.x, (image.height() + blockDim.y - 1) / blockDim.y);

    trueOutlinesRGBA(
        gridSize, blockSize, gridDim, blockDim, this->streamTrueOutLine,
        d_img, d_img_copy, image.width(), image.height(), nPixels, thresh
    );
    cudaStreamSynchronize(this->streamTrueOutLine);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
    cudaFree(d_img_copy);
}

void MainWindow::updatePosterizeThumbnail() {
    const float thresh = 255.0f / ui->posterizeThresh->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnPosterize);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;

    posterizeRGBA(gridSize, blockSize, this->streamPosterize, d_img, size, thresh);
    cudaStreamSynchronize(this->streamPosterize);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updateRadialBlurThumbnail() {
    const int blurRadius = ui->radialBlurRadius->value();
    const float intensity = static_cast<float>(ui->Intensity->value());
    const float centerX = static_cast<float>(ui->centerX->value());
    const float centerY = static_cast<float>(ui->centerY->value());

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnRadialBlur);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((image.width() + blockDim.x - 1) / blockDim.x, (image.height() + blockDim.y - 1) / blockDim.y);

    radialBlurRGBA(gridDim, blockDim, this->streamRadialBlur, d_img, image.width(), image.height(), centerX, centerY, blurRadius, intensity);
    cudaStreamSynchronize(this->streamRadialBlur);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updateCensorThumbnail() {
    const int pixelWidth = ui->censorWidth->value();
    const int pixelHeight = ui->censorHeight->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnCensor);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((image.width() + blockDim.x - 1) / blockDim.x, (image.height() + blockDim.y - 1) / blockDim.y);

    censorRGBA(gridDim, blockDim, this->streamCensor, d_img, image.width(), image.height(), pixelWidth, pixelHeight);
    cudaStreamSynchronize(this->streamCensor);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updatePixelateThumbnail() {
    const int pixelWidth = ui->pixelWidth->value();
    const int pixelHeight = ui->pixelHeight->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnPixelate);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((image.width() + blockDim.x - 1) / blockDim.x, (image.height() + blockDim.y - 1) / blockDim.y);

    pixelateRGBA(gridDim, blockDim, this->streamPixelate, d_img, image.width(), image.height(), pixelWidth, pixelHeight);
    cudaStreamSynchronize(this->streamPixelate);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
}

void MainWindow::updateVintage8bitThumbnail() {
    const int pixelWidth = ui->vintagePixelWidth->value();
    const int pixelHeight = ui->vintagePixelHeight->value();
    const float thresh = 255.0f / static_cast<float>(ui->vintagePosterizeThresh->value());

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnVintage8bit);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char colorsRGB[] = {
        67 , 9  , 64,
        133, 70 , 61,
        197, 131, 59,
        127, 124, 58,
        61 , 64 , 61,
        191, 188, 122,
        255, 194, 122,
        255, 246, 121,
        254, 251, 187,
        197, 134, 125,
        118, 72 , 56,
        250, 202, 120,
        205, 126, 47,
        105, 44 , 20
    };

    unsigned char* d_img;
    unsigned char* d_colorsRGB;
    cudaMalloc(&d_img, size);
    cudaMalloc(&d_colorsRGB, sizeof(colorsRGB));
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colorsRGB, colorsRGB, sizeof(colorsRGB), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((image.width() + blockDim.x - 1) / blockDim.x, (image.height() + blockDim.y - 1) / blockDim.y);

    int blockSize = 1024;
    int gridPixels = (nPixels + blockSize - 1) / blockSize;
    int gridSize = (size + blockSize - 1) / blockSize;

    vintage8bitRGBA(
        gridDim, blockDim,
        gridPixels, blockSize,
        gridSize, this->streamVintage8bit,
        d_img, pixelWidth, pixelHeight, thresh,
        d_colorsRGB, sizeof(colorsRGB) / 3,
        image.width(), image.height(), nPixels, size
    );
    cudaStreamSynchronize(this->streamVintage8bit);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    cudaFree(d_img);
    cudaFree(d_colorsRGB);
}

void MainWindow::updateChangePaletteThumbnail() {
    if (this->changePaletteColorsVector.empty()) return;

    unsigned char* colorsBGR = this->changePaletteColorsVector.data();
    unsigned char* colorsRGB = new unsigned char[this->changePaletteColorsVector.size()];
    int numColors = this->changePaletteColorsVector.size() / 3;

    for (int i = 0; i < numColors; ++i) {
        int index = i * 3;
        colorsRGB[index] = colorsBGR[index + 2]; // R
        colorsRGB[index + 1] = colorsBGR[index + 1]; // G
        colorsRGB[index + 2] = colorsBGR[index];     // B
    }

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnChangePalette);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    unsigned char* d_colorsRGB;
    cudaMalloc(&d_img, size);
    cudaMalloc(&d_colorsRGB, this->changePaletteColorsVector.size() * sizeof(unsigned char));
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colorsRGB, colorsRGB, this->changePaletteColorsVector.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (nPixels + blockSize - 1) / blockSize;

    changePaletteRGBA(gridSize, blockSize, this->streamChangePalette, d_img, nPixels, d_colorsRGB, numColors);
    cudaStreamSynchronize(this->streamChangePalette);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    delete[] colorsRGB;
    cudaFree(d_img);
    cudaFree(d_colorsRGB);
}

void MainWindow::updateMonoMaskThumbnail() {
    if (this->monoMaskColorsVector.size() < 6) return;

    unsigned char* colorsBGR = this->monoMaskColorsVector.data();
    unsigned char* colorsRGB = new unsigned char[this->monoMaskColorsVector.size()];
    int numColors = this->monoMaskColorsVector.size() / 3;

    for (int i = 0; i < numColors; ++i) {
        int index = i * 3;
        colorsRGB[index] = colorsBGR[index + 2]; // R
        colorsRGB[index + 1] = colorsBGR[index + 1]; // G
        colorsRGB[index + 2] = colorsBGR[index];     // B
    }

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnMonoMask);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    unsigned char* img = new unsigned char[size];
    memcpy(img, image.constBits(), size);

    unsigned char* d_img;
    unsigned char* d_colorsRGB;
    cudaMalloc(&d_img, size);
    cudaMalloc(&d_colorsRGB, this->monoMaskColorsVector.size() * sizeof(unsigned char));
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colorsRGB, colorsRGB, this->monoMaskColorsVector.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (nPixels + blockSize - 1) / blockSize;

    monoMaskRGBA(gridSize, blockSize, this->streamMonoMask, d_img, nPixels, d_colorsRGB, numColors);
    cudaStreamSynchronize(this->streamMonoMask);

    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);

    QImage result(img, image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);

    // Cleanup
    delete[] img;
    delete[] colorsRGB;
    cudaFree(d_img);
    cudaFree(d_colorsRGB);
}


static void setThumbnail(QPushButton* button, QPixmap pixmap) {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(button);
    if (!effectBtn) return;

    effectBtn->setThumbnail(pixmap);
}

void MainWindow::newThumbnails() {
    QPixmap pixmap(QString::fromStdWString(this->selectedFilePath));
    if (pixmap.isNull()) return;

    // Scale down with smooth transformation
    this->selectedPixmap = pixmap.scaled(960, 540, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    setThumbnail(ui->btnHueShift, this->selectedPixmap);
    setThumbnail(ui->btnFilter, this->selectedPixmap);
    setThumbnail(ui->btnBinary, this->selectedPixmap);
    setThumbnail(ui->btnInverseColors, this->selectedPixmap);
    setThumbnail(ui->btnInverseContrast, this->selectedPixmap);
    setThumbnail(ui->btnMonoChrome, this->selectedPixmap);
    setThumbnail(ui->btnBlur, this->selectedPixmap);
    setThumbnail(ui->btnOutlines, this->selectedPixmap);
    setThumbnail(ui->btnTrueOutlines, this->selectedPixmap);
    setThumbnail(ui->btnPosterize, this->selectedPixmap);
    setThumbnail(ui->btnRadialBlur, this->selectedPixmap);
    setThumbnail(ui->btnCensor, this->selectedPixmap);
    setThumbnail(ui->btnPixelate, this->selectedPixmap);
    setThumbnail(ui->btnVintage8bit, this->selectedPixmap);
    setThumbnail(ui->btnChangePalette, this->selectedPixmap);
    setThumbnail(ui->btnMonoMask, this->selectedPixmap);

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnRadialBlur);
    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    ui->centerX->setMaximum(originalPixmap.width());
    ui->centerY->setMaximum(originalPixmap.height());

    ui->centerX->setValue(0.5f * originalPixmap.width());
    ui->centerY->setValue(0.5f * originalPixmap.height());

    effectBtn = qobject_cast<EffectButton*>(ui->btnCensor);
    originalPixmap = effectBtn->getOriginalPixmap();

    ui->censorWidth->setMaximum(originalPixmap.width());
    ui->censorHeight->setMaximum(originalPixmap.height());

    ui->pixelWidth->setMaximum(originalPixmap.width());
    ui->pixelHeight->setMaximum(originalPixmap.height());

    ui->vintagePixelWidth->setMaximum(originalPixmap.width());
    ui->vintagePixelHeight->setMaximum(originalPixmap.height());


    this->updateThumbnails();
}

void MainWindow::updateThumbnails() {
    this->updateHueShiftThumbnail();
    this->updateFilterThumbnail();
    this->updateBinaryThumbnail();
    this->updateInverseColorsThumbnail();
    this->updateInverseContrastThumbnail();
    this->updateMonoChromeThumbnail();
    this->updateBlurThumbnail();
    this->updateOutLineThumbnail();
    this->updateTrueOutLineThumbnail();
    this->updatePosterizeThumbnail();
    this->updateRadialBlurThumbnail();
    this->updateCensorThumbnail();
    this->updatePixelateThumbnail();
    this->updateVintage8bitThumbnail();
    this->updateChangePaletteThumbnail();
    this->updateMonoMaskThumbnail();
}
