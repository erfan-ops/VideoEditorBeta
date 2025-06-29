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
#include "utils.h"

#include "EffectButton.h"
#include "colorButton.h"

#include "globals.h"
#include "baseProcessor.h"


MainWindow::MainWindow(int argc, char** argv, QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/icon.ico"));
    setWindowTitle(QString("RetroShade"));

    // create OpenCL context and queue if CUDA isn't available
    if (!isCudaAvailable()) {
        globalContextOpenCL = openclUtils::createContext(globalDeviceOpenCL);
        globalQueueOpenCL = openclUtils::createCommandQueue(globalContextOpenCL, globalDeviceOpenCL);
    }

    // Then initialize the processor
    BaseProcessor::init();
    SoftPaletteProcessor::init();
    BlackAndWhiteProcessor::init();
    BlurProcessor::init();
    CensorProcessor::init();
    FlatLightProcessor::init();
    ChangePaletteProcessor::init();
    HueShiftProcessor::init();
    InverseColorsProcessor::init();
    InverseContrastProcessor::init();
    LensFilterProcessor::init();
    MonoChromeProcessor::init();
    MonoMaskProcessor::init();
    PosterizeProcessor::init();
    PixelateProcessor::init();
    OutlinesProcessor::init();
    MagicEyeProcessor::init();
    TrueOutlinesProcessor::init();
    RadialBlurProcessor::init();
    Vintage8BitProcessor::init();
    FlatSaturationProcessor::init();

    // Connect signals
    QObject::connect(ui->btnSelect, &QPushButton::clicked, [&]() {
        std::wstring newPath = FileDialog::OpenFileDialog(L"All Files");
        if (!newPath.empty()) {
            selectedFilePath = newPath;
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
    replaceButtonWithEffectButton(ui->btnSoftPalette, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnFlatLight, ":/samples/samples/sample.jpg");
    replaceButtonWithEffectButton(ui->btnFlatSaturation, ":/samples/samples/sample.jpg");

    if (argc > 1 && std::filesystem::exists(argv[1])) {
        std::wstring argvPath = stringUtils::string_to_wstring(argv[1]);
        if (videoExtentions.find(fileUtils::splitextw(argvPath).second) != videoExtentions.end()) {
            selectedFilePath = argvPath;
            ui->label->setText(QString::fromStdWString(L"Selected: " + selectedFilePath));
        }
        else if (imageExtentions.find(fileUtils::splitextw(argvPath).second) != imageExtentions.end()) {
            selectedFilePath = argvPath;
            ui->label->setText(QString::fromStdWString(L"Selected: " + selectedFilePath));
            this->newThumbnails();
        }
    }

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
        float hue = ui->hueShiftSlider->value() / 100.0f;
        float saturation = ui->saturationSlider->value() / 100.0f;
        float lightness = ui->lighnessSlider->value() / 100.0f;

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VHueShiftWorker(hue, saturation, lightness);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IHueShiftWorker(hue, saturation, lightness);
        }

        processEffect(ui->btnHueShift, worker);
        });
    QObject::connect(ui->hueShiftSlider, QOverload<int>::of(&QSlider::valueChanged), this, &MainWindow::updateHueShiftThumbnail);
    QObject::connect(ui->saturationSlider, QOverload<int>::of(&QSlider::valueChanged), this, &MainWindow::updateHueShiftThumbnail);
    QObject::connect(ui->lighnessSlider, QOverload<int>::of(&QSlider::valueChanged), this, &MainWindow::updateHueShiftThumbnail);

    QObject::connect(ui->btnRadialBlur, &QPushButton::clicked, this, [&]() {
        int blurRadius = ui->radialBlurRadius->value();
        float intensity = static_cast<float>(ui->Intensity->value());
        float centerX = static_cast<float>(ui->centerX->value()) / this->widthRatio;
        float centerY = static_cast<float>(ui->centerY->value()) / this->heightRatio;

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
            int index = this->changePaletteColorButtons.size();

            ColorButton* colorButton = new ColorButton(color, index);
            colorButton->setFixedSize(162, 20);
            colorButton->setStyleSheet(QString(
                "background-color: %1; "
                "border-radius: 4px; "
                "border: 2px solid transparent;"
            ).arg(color.name()));

            changePaletteColorButtons.push_back(colorButton);

            QObject::connect(colorButton, &QPushButton::clicked, this, [=]() {
                for (ColorButton* btn : this->changePaletteColorButtons) {
                    btn->setStyleSheet(btn->styleSheet().replace("border: 2px solid #0078d7;", "border: 2px solid transparent;"));
                    btn->setSelected(false);
                }

                colorButton->setStyleSheet(colorButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
                this->changePaletteSelectedColor = colorButton->getIndex();
                colorButton->setSelected(true);
                });

            // Add to the layout of the scroll area's contents widget
            QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->changePaletteScrollAreaWidgetContents->layout());
            if (layout) {
                layout->addWidget(colorButton);

                // Auto-scroll to bottom
                ui->changePaletteScrollArea->verticalScrollBar()->setValue(
                    ui->changePaletteScrollArea->verticalScrollBar()->maximum()
                );
            }

            this->updateChangePaletteThumbnail();
        }
        });
    QObject::connect(ui->changePaletteEditColorBtn, &QPushButton::clicked, this, [=]() {
        int selectedIndex = this->changePaletteSelectedColor;
        if (selectedIndex < 0 || selectedIndex >= this->changePaletteColorButtons.size())
            return;

        ColorButton* colorButton = this->changePaletteColorButtons[selectedIndex];
        QColor currentColor = colorButton->getColor();

        QColor newColor = QColorDialog::getColor(currentColor, this, "Select a Color");
        if (newColor.isValid()) {
            // Update button appearance
            colorButton->setColor(newColor);

            // Update color vector (stored as B, G, R)
            int vectorIndex = selectedIndex * 3;
            if (vectorIndex + 2 < this->changePaletteColorsVector.size()) {
                this->changePaletteColorsVector[vectorIndex] = newColor.blue();
                this->changePaletteColorsVector[vectorIndex + 1] = newColor.green();
                this->changePaletteColorsVector[vectorIndex + 2] = newColor.red();
            }

            this->updateChangePaletteThumbnail();
        }
        });
    QObject::connect(ui->changePaletteRemoveColorBtn, &QPushButton::clicked, this, [=]() {
        int selectedIndex = this->changePaletteSelectedColor;
        if (selectedIndex < 0 || selectedIndex >= this->changePaletteColorButtons.size())
            return;

        // Remove RGB values (3 components per color)
        int vectorIndex = selectedIndex * 3;
        if (vectorIndex + 2 < this->changePaletteColorsVector.size()) {
            this->changePaletteColorsVector.erase(
                this->changePaletteColorsVector.begin() + vectorIndex,
                this->changePaletteColorsVector.begin() + vectorIndex + 3
            );
        }

        // Remove the button from layout and delete it
        QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->changePaletteScrollAreaWidgetContents->layout());
        ColorButton* buttonToRemove = this->changePaletteColorButtons[selectedIndex];
        if (layout && buttonToRemove) {
            layout->removeWidget(buttonToRemove);
            buttonToRemove->deleteLater();
        }

        // Remove from button list
        this->changePaletteColorButtons.removeAt(selectedIndex);

        // Update indices of remaining buttons
        for (int i = selectedIndex; i < this->changePaletteColorButtons.size(); ++i) {
            this->changePaletteColorButtons[i]->setIndex(i);
        }

        // Reset selection
        this->changePaletteSelectedColor -= 1;

        if (this->changePaletteSelectedColor >= 0) {
            ColorButton* previousButton = this->changePaletteColorButtons[this->changePaletteSelectedColor];
            previousButton->setStyleSheet(previousButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
        }
        else if (!this->changePaletteColorButtons.empty()) {
            this->changePaletteSelectedColor = 0;
            ColorButton* previousButton = this->changePaletteColorButtons[this->changePaletteSelectedColor];
            previousButton->setStyleSheet(previousButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
        }

        this->updateChangePaletteThumbnail();
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
            int index = this->monoMaskColorButtons.size();

            ColorButton* colorButton = new ColorButton(color, index);
            colorButton->setFixedSize(162, 20);
            colorButton->setStyleSheet(QString(
                "background-color: %1; "
                "border-radius: 4px; "
                "border: 2px solid transparent;"
            ).arg(color.name()));

            monoMaskColorButtons.push_back(colorButton);

            QObject::connect(colorButton, &QPushButton::clicked, this, [=]() {
                for (ColorButton* btn : this->monoMaskColorButtons) {
                    btn->setStyleSheet(btn->styleSheet().replace("border: 2px solid #0078d7;", "border: 2px solid transparent;"));
                    btn->setSelected(false);
                }

                colorButton->setStyleSheet(colorButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
                this->monoMaskSelectedColor = colorButton->getIndex();
                colorButton->setSelected(true);
                });

            // Add to the layout of the scroll area's contents widget
            QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->monoMaskScrollAreaWidgetContents->layout());
            if (layout) {
                layout->addWidget(colorButton);

                // Auto-scroll to bottom
                ui->monoMaskScrollArea->verticalScrollBar()->setValue(
                    ui->monoMaskScrollArea->verticalScrollBar()->maximum()
                );
            }

            this->updateMonoMaskThumbnail();
        }
        });
    QObject::connect(ui->monoMaskEditColorBtn, &QPushButton::clicked, this, [=]() {
        int selectedIndex = this->monoMaskSelectedColor;
        if (selectedIndex < 0 || selectedIndex >= this->monoMaskColorButtons.size())
            return;

        ColorButton* colorButton = this->monoMaskColorButtons[selectedIndex];
        QColor currentColor = colorButton->getColor();

        QColor newColor = QColorDialog::getColor(currentColor, this, "Select a Color");
        if (newColor.isValid()) {
            // Update button appearance
            colorButton->setColor(newColor);

            // Update color vector (stored as B, G, R)
            int vectorIndex = selectedIndex * 3;
            if (vectorIndex + 2 < this->monoMaskColorsVector.size()) {
                this->monoMaskColorsVector[vectorIndex] = newColor.blue();
                this->monoMaskColorsVector[vectorIndex + 1] = newColor.green();
                this->monoMaskColorsVector[vectorIndex + 2] = newColor.red();
            }

            this->updateMonoMaskThumbnail();
        }
        });
    QObject::connect(ui->monoMaskRemoveColorBtn, &QPushButton::clicked, this, [=]() {
        int selectedIndex = this->monoMaskSelectedColor;
        if (selectedIndex < 0 || selectedIndex >= this->monoMaskColorButtons.size())
            return;

        // Remove RGB values (3 components per color)
        int vectorIndex = selectedIndex * 3;
        if (vectorIndex + 2 < this->monoMaskColorsVector.size()) {
            this->monoMaskColorsVector.erase(
                this->monoMaskColorsVector.begin() + vectorIndex,
                this->monoMaskColorsVector.begin() + vectorIndex + 3
            );
        }

        // Remove the button from layout and delete it
        QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->monoMaskScrollAreaWidgetContents->layout());
        ColorButton* buttonToRemove = this->monoMaskColorButtons[selectedIndex];
        if (layout && buttonToRemove) {
            layout->removeWidget(buttonToRemove);
            buttonToRemove->deleteLater();
        }

        // Remove from button list
        this->monoMaskColorButtons.removeAt(selectedIndex);

        // Update indices of remaining buttons
        for (int i = selectedIndex; i < this->monoMaskColorButtons.size(); ++i) {
            this->monoMaskColorButtons[i]->setIndex(i);
        }

        // Reset selection
        this->monoMaskSelectedColor -= 1;

        if (this->monoMaskSelectedColor >= 0) {
            ColorButton* previousButton = this->monoMaskColorButtons[this->monoMaskSelectedColor];
            previousButton->setStyleSheet(previousButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
        }
        else if (!this->monoMaskColorButtons.empty()) {
            this->monoMaskSelectedColor = 0;
            ColorButton* previousButton = this->monoMaskColorButtons[this->monoMaskSelectedColor];
            previousButton->setStyleSheet(previousButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
        }

        this->updateMonoMaskThumbnail();
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

    if (!ui->softPaletteScrollAreaWidgetContents->layout()) {
        QVBoxLayout* layout = new QVBoxLayout(ui->softPaletteScrollAreaWidgetContents);
        layout->setSizeConstraint(QLayout::SetMinAndMaxSize);
        layout->setAlignment(Qt::AlignTop);
        ui->softPaletteScrollAreaWidgetContents->setLayout(layout);
    }

    QObject::connect(ui->softPaletteAddColorBtn, &QPushButton::clicked, this, [&]() {
        QColor color = QColorDialog::getColor(Qt::white, this, "Select a Color");
        if (color.isValid()) {
            this->softPaletteColorsVector.push_back(color.blue());
            this->softPaletteColorsVector.push_back(color.green());
            this->softPaletteColorsVector.push_back(color.red());

            // Create a styled label
            int index = this->softPaletteColorButtons.size();

            ColorButton* colorButton = new ColorButton(color, index);
            colorButton->setFixedSize(162, 20);
            colorButton->setStyleSheet(QString(
                "background-color: %1; "
                "border-radius: 4px; "
                "border: 2px solid transparent;"
            ).arg(color.name()));

            softPaletteColorButtons.push_back(colorButton);

            QObject::connect(colorButton, &QPushButton::clicked, this, [=]() {
                for (ColorButton* btn : this->softPaletteColorButtons) {
                    btn->setStyleSheet(btn->styleSheet().replace("border: 2px solid #0078d7;", "border: 2px solid transparent;"));
                    btn->setSelected(false);
                }

                colorButton->setStyleSheet(colorButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
                this->softPaletteSelectedColor = colorButton->getIndex();
                colorButton->setSelected(true);
                });

            // Add to the layout of the scroll area's contents widget
            QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->softPaletteScrollAreaWidgetContents->layout());
            if (layout) {
                layout->addWidget(colorButton);

                // Auto-scroll to bottom
                ui->softPaletteScrollArea->verticalScrollBar()->setValue(
                    ui->softPaletteScrollArea->verticalScrollBar()->maximum()
                );
            }

            this->updateSoftPaletteThumbnail();
        }
        });
    QObject::connect(ui->softPaletteEditColorBtn, &QPushButton::clicked, this, [=]() {
        int selectedIndex = this->softPaletteSelectedColor;
        if (selectedIndex < 0 || selectedIndex >= this->softPaletteColorButtons.size())
            return;

        ColorButton* colorButton = this->softPaletteColorButtons[selectedIndex];
        QColor currentColor = colorButton->getColor();

        QColor newColor = QColorDialog::getColor(currentColor, this, "Select a Color");
        if (newColor.isValid()) {
            // Update button appearance
            colorButton->setColor(newColor);

            // Update color vector (stored as B, G, R)
            int vectorIndex = selectedIndex * 3;
            if (vectorIndex + 2 < this->softPaletteColorsVector.size()) {
                this->softPaletteColorsVector[vectorIndex] = newColor.blue();
                this->softPaletteColorsVector[vectorIndex + 1] = newColor.green();
                this->softPaletteColorsVector[vectorIndex + 2] = newColor.red();
            }

            this->updateSoftPaletteThumbnail();
        }
        });
    QObject::connect(ui->softPaletteRemoveColorBtn, &QPushButton::clicked, this, [=]() {
        int selectedIndex = this->softPaletteSelectedColor;
        if (selectedIndex < 0 || selectedIndex >= this->softPaletteColorButtons.size())
            return;

        // Remove RGB values (3 components per color)
        int vectorIndex = selectedIndex * 3;
        if (vectorIndex + 2 < this->softPaletteColorsVector.size()) {
            this->softPaletteColorsVector.erase(
                this->softPaletteColorsVector.begin() + vectorIndex,
                this->softPaletteColorsVector.begin() + vectorIndex + 3
            );
        }

        // Remove the button from layout and delete it
        QVBoxLayout* layout = qobject_cast<QVBoxLayout*>(ui->softPaletteScrollAreaWidgetContents->layout());
        ColorButton* buttonToRemove = this->softPaletteColorButtons[selectedIndex];
        if (layout && buttonToRemove) {
            layout->removeWidget(buttonToRemove);
            buttonToRemove->deleteLater();
        }

        // Remove from button list
        this->softPaletteColorButtons.removeAt(selectedIndex);

        // Update indices of remaining buttons
        for (int i = selectedIndex; i < this->softPaletteColorButtons.size(); ++i) {
            this->softPaletteColorButtons[i]->setIndex(i);
        }

        // Reset selection
        this->softPaletteSelectedColor -= 1;

        if (this->softPaletteSelectedColor >= 0) {
            ColorButton* previousButton = this->softPaletteColorButtons[this->softPaletteSelectedColor];
            previousButton->setStyleSheet(previousButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
        }
        else if (!this->softPaletteColorButtons.empty()) {
            this->softPaletteSelectedColor = 0;
            ColorButton* previousButton = this->softPaletteColorButtons[this->softPaletteSelectedColor];
            previousButton->setStyleSheet(previousButton->styleSheet().replace("border: 2px solid transparent;", "border: 2px solid #0078d7;"));
        }

        this->updateSoftPaletteThumbnail();
        });
    QObject::connect(ui->btnSoftPalette, &QPushButton::clicked, this, [&]() {
        unsigned char* colorsBGR = this->softPaletteColorsVector.data();
        int numColors = this->softPaletteColorsVector.size() / 3;

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VSoftPaletteWorker(colorsBGR, numColors);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new ISoftPaletteWorker(colorsBGR, numColors);
        }

        processEffect(ui->btnSoftPalette, worker);
        });
    
    QObject::connect(ui->btnFlatLight, &QPushButton::clicked, this, [&]() {
        float lightness = ui->flatLightnessSlider->value() / 100.0f;

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VFlatLightWorker(lightness);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IFlatLightWorker(lightness);
        }

        processEffect(ui->btnFlatLight, worker);
        });
    QObject::connect(ui->flatLightnessSlider, QOverload<int>::of(&QSlider::valueChanged), this, &MainWindow::updateFlatLightThumbnail);

    QObject::connect(ui->btnFlatSaturation, &QPushButton::clicked, this, [&]() {
        float saturation = ui->flatSaturationSlider->value() / 100.f;

        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VFlatSaturationWorker(saturation);
            QObject::connect(worker, &EffectBase::progressChanged, this, &MainWindow::updateProgress, Qt::QueuedConnection);
        }
        else {
            worker = new IFlatSaturationWorker(saturation);
        }

        processEffect(ui->btnFlatSaturation, worker);
        });
    QObject::connect(ui->flatSaturationSlider, QOverload<int>::of(&QSlider::valueChanged), this, &MainWindow::updateFlatSaturationThumbnail);
}

MainWindow::~MainWindow()
{
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
    float hue = ui->hueShiftSlider->value() / 100.0f;
    float saturation = ui->saturationSlider->value() / 100.0f;
    float lightness = ui->lighnessSlider->value() / 100.0f;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnHueShift);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    HueShiftProcessor processor(size, nPixels, hue, saturation, lightness);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateFilterThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnFilter);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    float passThreshValues[] = { filterColor.redF(), filterColor.greenF(), filterColor.blueF(), 1.0f };

    LensFilterProcessor processor(size, passThreshValues, 4);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
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

    BlackAndWhiteProcessor blackAndWhiteProcessor(nPixels, size, middle);

    blackAndWhiteProcessor.setImage(image.bits(), size);
    blackAndWhiteProcessor.processRGBA();
    memcpy(image.bits(), blackAndWhiteProcessor.getImage(), size);

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateInverseColorsThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnInverseColors);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    InverseColorsProcessor processor(size);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateInverseContrastThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnInverseContrast);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    InverseContrastProcessor processor(size, nPixels);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateMonoChromeThumbnail() {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnMonoChrome);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();
    int nPixels = image.width() * image.height();

    MonoChromeProcessor processor(size, nPixels);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateBlurThumbnail() {
    int radius = ui->blurRadius->value() * this->widthRatio;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnBlur);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    BlurProcessor blurProcessor(size, image.width(), image.height(), radius);

    blurProcessor.setImage(image.bits());
    blurProcessor.processRGBA();
    memcpy(image.bits(), blurProcessor.getImage(), size);

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateOutLineThumbnail() {
    const int thicknessX = ui->ThicknessX->value() * this->widthRatio;
    const int thicknessY = ui->ThicknessY->value() * this->heightRatio;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnOutlines);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    OutlinesProcessor processor(size, image.width(), image.height(), thicknessX, thicknessY);

    processor.upload(image.constBits());
    processor.processRGBA();
    processor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateTrueOutLineThumbnail() {
    const int thresh = ui->trueOutlinesThresh->value() * this->widthRatio;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnTrueOutlines);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    TrueOutlinesProcessor processor(image.sizeInBytes(), image.width(), image.height(), thresh);

    processor.upload(image.constBits());
    processor.processRGBA();
    processor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updatePosterizeThumbnail() {
    const float thresh = 255.0f / ui->posterizeThresh->value();

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnPosterize);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    int size = image.sizeInBytes();

    PosterizeProcessor processor(size, thresh);

    processor.upload(image.constBits());
    processor.processRGBA();
    processor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
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

    RadialBlurProcessor processor(image.sizeInBytes(), image.width(), image.height(), centerX, centerY, blurRadius, intensity);

    processor.upload(image.constBits());
    processor.processRGBA();
    processor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateCensorThumbnail() {
    const int pixelWidth = ui->censorWidth->value() * this->widthRatio;
    const int pixelHeight = ui->censorHeight->value() * this->heightRatio;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnCensor);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();

    CensorProcessor censorProcessor(size, image.width(), image.height(), pixelWidth, pixelHeight);

    censorProcessor.setImage(image.constBits());
    censorProcessor.processRGBA();
    censorProcessor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updatePixelateThumbnail() {
    const int pixelWidth = ui->pixelWidth->value() * this->widthRatio;
    const int pixelHeight = ui->pixelHeight->value() * this->heightRatio;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnPixelate);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    PixelateProcessor processor(image.sizeInBytes(), image.width(), image.height(), pixelWidth, pixelHeight);

    processor.upload(image.constBits());
    processor.processRGBA();
    processor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateVintage8bitThumbnail() {
    const int pixelWidth = ui->vintagePixelWidth->value() * this->widthRatio;
    const int pixelHeight = ui->vintagePixelHeight->value() * this->heightRatio;
    const float thresh = 255.0f / static_cast<float>(ui->vintagePosterizeThresh->value());

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnVintage8bit);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    Vintage8BitProcessor processor(image.sizeInBytes(), image.width(), image.height(), pixelWidth, pixelHeight, thresh);

    processor.upload(image.constBits());
    processor.processRGBA();
    processor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateChangePaletteThumbnail() {
    if (this->changePaletteColorsVector.empty()) return;

    unsigned char* colorsBGR = this->changePaletteColorsVector.data();
    int numColors = this->changePaletteColorsVector.size() / 3;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnChangePalette);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    ChangePaletteProcessor processor(size, nPixels, colorsBGR, numColors);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateMonoMaskThumbnail() {
    if (this->monoMaskColorsVector.size() < 6) return;

    unsigned char* colorsBGR = this->monoMaskColorsVector.data();
    int numColors = this->monoMaskColorsVector.size() / 3;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnMonoMask);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    // 2. Process the image through CUDA
    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    MonoMaskProcessor processor(nPixels, size, colorsBGR, numColors);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateSoftPaletteThumbnail() {
    if (this->softPaletteColorsVector.empty()) return;

    unsigned char* colorsBGR = this->softPaletteColorsVector.data();
    int numColors = this->softPaletteColorsVector.size() / 3;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnSoftPalette);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    SoftPaletteProcessor softPaletteProcessor(nPixels, size, colorsBGR, numColors);

    softPaletteProcessor.upload(image.constBits());
    softPaletteProcessor.processRGBA();
    softPaletteProcessor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateFlatLightThumbnail() {
    float lightness = ui->flatLightnessSlider->value() / 100.0f;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnFlatLight);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    FlatLightProcessor processor(size, nPixels, lightness);

    processor.setImage(image.constBits());
    processor.processRGBA();
    processor.upload(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}

void MainWindow::updateFlatSaturationThumbnail() {
    float saturation = ui->flatSaturationSlider->value() / 100.f;

    EffectButton* effectBtn = qobject_cast<EffectButton*>(ui->btnFlatSaturation);
    if (!effectBtn) return;

    QPixmap originalPixmap = effectBtn->getOriginalPixmap();

    QImage image = originalPixmap.toImage().convertToFormat(QImage::Format_RGBA8888);

    const int size = image.sizeInBytes();
    const int nPixels = image.width() * image.height();

    FlatSaturationProcessor processor(size, nPixels, saturation);

    processor.upload(image.constBits());
    processor.processRGBA();
    processor.download(image.bits());

    QImage result(image.bits(), image.width(), image.height(), QImage::Format_RGBA8888);
    QPixmap resultPixmap = QPixmap::fromImage(result);

    effectBtn->setProcessedPixmap(resultPixmap);
}


static void setThumbnail(QPushButton* button, QPixmap pixmap) {
    EffectButton* effectBtn = qobject_cast<EffectButton*>(button);
    if (!effectBtn) return;

    effectBtn->setThumbnail(pixmap);
}

void MainWindow::newThumbnails() {
    QPixmap pixmap(QString::fromStdWString(this->selectedFilePath));
    if (pixmap.isNull()) return;

    int orgW = pixmap.width();
    int orgH = pixmap.height();

    // Scale down with smooth transformation
    this->selectedPixmap = pixmap.scaled(960, 540, Qt::AspectRatioMode::KeepAspectRatio, Qt::TransformationMode::SmoothTransformation);

    this->widthRatio = static_cast<float>(this->selectedPixmap.width()) / static_cast<float>(orgW);
    this->heightRatio = static_cast<float>(this->selectedPixmap.height()) / static_cast<float>(orgH);

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
    setThumbnail(ui->btnSoftPalette, this->selectedPixmap);
    setThumbnail(ui->btnFlatLight, this->selectedPixmap);
    setThumbnail(ui->btnFlatSaturation, this->selectedPixmap);

    ui->centerX->setMaximum(this->selectedPixmap.width());
    ui->centerY->setMaximum(this->selectedPixmap.height());

    ui->centerX->setValue(0.5f * this->selectedPixmap.width());
    ui->centerY->setValue(0.5f * this->selectedPixmap.height());

    int widthOver80 = orgW / 80;

    ui->censorWidth->setValue(widthOver80);
    ui->censorHeight->setValue(widthOver80);

    ui->censorWidth->setMaximum(orgW);
    ui->censorHeight->setMaximum(orgH);

    ui->pixelWidth->setValue(widthOver80);
    ui->pixelHeight->setValue(widthOver80);

    ui->pixelWidth->setMaximum(orgW);
    ui->pixelHeight->setMaximum(orgH);

    ui->vintagePixelWidth->setValue(widthOver80);
    ui->vintagePixelHeight->setValue(widthOver80);

    ui->vintagePixelWidth->setMaximum(orgW);
    ui->vintagePixelHeight->setMaximum(orgH);

    ui->blurRadius->setValue(widthOver80);
    ui->blurRadius->setMaximum(orgW / 10);

    float rWidthRatio = 1.f / this->widthRatio;

    ui->ThicknessX->setValue(static_cast<int>(std::ceil(rWidthRatio)));
    ui->ThicknessY->setValue(static_cast<int>(std::ceil(rWidthRatio)));
    ui->trueOutlinesThresh->setValue(static_cast<int>(std::ceil(rWidthRatio * 4.f)));


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
    this->updateSoftPaletteThumbnail();
    this->updateFlatLightThumbnail();
    this->updateFlatSaturationThumbnail();
}
