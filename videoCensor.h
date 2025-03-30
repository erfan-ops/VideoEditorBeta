#pragma once

#include "videoEffectBase.h"

class VCensorWorker : public VideoEffect {
    Q_OBJECT
public:
    VCensorWorker(int pixelWidth, int pixelHeight, QObject* parent = nullptr) : VideoEffect(parent), m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight) {}
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
};
