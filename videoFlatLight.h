#pragma once

#include "videoEffectBase.h"

class VFlatLightWorker : public VideoEffect {
    Q_OBJECT
public:
    VFlatLightWorker(float lightness, QObject* parent = nullptr) : VideoEffect(parent), m_lightness(lightness) {}
    void process() override;
private:
    float m_lightness;
};
