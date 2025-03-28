#pragma once

#include "videoEffectBase.h"

class VCensorWorker : public VideoEffect {
    Q_OBJECT
public:
    VCensorWorker(int pixelWidht, int pixelHeight, QObject* parent = nullptr);
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
};
