#pragma once

#include "videoEffectBase.h"

class VPixelateWorker : public VideoEffect {
    Q_OBJECT
public:
    VPixelateWorker(int pixelWidht, int pixelHeight, QObject* parent = nullptr);
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
};
