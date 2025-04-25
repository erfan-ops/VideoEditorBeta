#pragma once

#include "imageEffectBase.h"

class IVintage8bitWorker : public ImageEffect {
    Q_OBJECT
public:
    IVintage8bitWorker(int pixelWidth, int pixelHeight, int thresh, QObject* parent = nullptr)
        : m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight), m_thresh(255.0f / thresh) {};
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
    float m_thresh;
};
