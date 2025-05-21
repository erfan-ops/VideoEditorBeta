#pragma once

#include "imageEffectBase.h"

class IPixelateWorker : public ImageEffect {
    Q_OBJECT
public:
    IPixelateWorker(int pixelWidht, int pixelHeight, QObject* parent = nullptr)
        : m_pixelWidth(pixelWidht), m_pixelHeight(pixelHeight) {}
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
};
