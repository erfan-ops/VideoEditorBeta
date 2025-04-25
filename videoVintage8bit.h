#pragma once

#include "videoEffectBase.h"


class VVintage8bitWorker : public VideoEffect {
    Q_OBJECT
public:
    VVintage8bitWorker(int pixelWidth, int pixelHeight, int thresh, QObject* parent = nullptr)
        : m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight), m_thresh(255.0f / thresh) {};
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
    float m_thresh;
};
