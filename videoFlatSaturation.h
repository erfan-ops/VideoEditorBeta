#pragma once

#include "videoEffectBase.h"

class VFlatSaturationWorker : public VideoEffect {
    Q_OBJECT
public:
    VFlatSaturationWorker(float saturation, QObject* parent = nullptr) : VideoEffect(parent), m_saturation(saturation) {}
    void process() override;
private:
    float m_saturation;
};
