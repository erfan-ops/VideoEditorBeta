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

#include "colorButton.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(int argc, char** argv, QWidget* parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow* ui;
    std::wstring selectedFilePath;
    std::wstring saveFilePath;
    QColor filterColor = QColor(255, 255, 255);
    QPixmap selectedPixmap;

    std::vector<unsigned char> changePaletteColorsVector;
    QVector<ColorButton*> changePaletteColorButtons;
    int changePaletteSelectedColor;
    
    std::vector<unsigned char> monoMaskColorsVector;
    QVector<ColorButton*> monoMaskColorButtons;
    int monoMaskSelectedColor;

    std::vector<unsigned char> softPaletteColorsVector;
    QVector<ColorButton*> softPaletteColorButtons;
    int softPaletteSelectedColor;

    float widthRatio{ 1 };
    float heightRatio{ 1 };

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
    void updateSoftPaletteThumbnail();
    void updateFlatLightThumbnail();
    void updateFlatSaturationThumbnail();
};
