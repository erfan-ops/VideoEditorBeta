#pragma once

#include <QMainWindow>
#include <QPushButton>
#include <QVector>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "effectBase.h" 
#include "timer.h"
#include "video.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow* ui;
    std::wstring selectedFilePath;
    std::wstring saveFilePath;
    QColor filterColor = QColor(255, 255, 255);
    QPixmap selectedPixmap;

    std::vector<unsigned char> changePaletteColorsVector;
    
    std::vector<unsigned char> monoMaskColorsVector;

    // effect specifics
    cudaStream_t streamHueShift;
    cudaStream_t streamFilter;
    cudaStream_t streamBinary;
    cudaStream_t streamInverseColors;
    cudaStream_t streamInverseContrast;
    cudaStream_t streamMonoChrome;
    cudaStream_t streamBlur;
    cudaStream_t streamOutLine;
    cudaStream_t streamTrueOutLine;
    cudaStream_t streamPosterize;
    cudaStream_t streamRadialBlur;
    cudaStream_t streamCensor;
    cudaStream_t streamPixelate;
    cudaStream_t streamVintage8bit;
    cudaStream_t streamChangePalette;
    cudaStream_t streamMonoMask;

    void processEffect(QPushButton* button, EffectBase* worker);
    void replaceButtonWithEffectButton(QPushButton*& button, const QString& imagePath);
    void updateProgress(const Video& video, const Timer& timer);
    void newThumbnails();
    void updateThumbnails();

    // effects
    void updateHueShiftThumbnail();
    void updateFilterThumbnail();
    void updateBinaryThumbnail();
    void updateInverseColorsThumbnail();
    void updateInverseContrastThumbnail();
    void updateMonoChromeThumbnail();
    void updateBlurThumbnail();
    void updateOutLineThumbnail();
    void updateTrueOutLineThumbnail();
    void updatePosterizeThumbnail();
    void updateRadialBlurThumbnail();
    void updateCensorThumbnail();
    void updatePixelateThumbnail();
    void updateVintage8bitThumbnail();
    void updateChangePaletteThumbnail();
    void updateMonoMaskThumbnail();
};
