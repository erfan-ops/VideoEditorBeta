#pragma once

#include "videoEffectBase.h"


class VHueShiftWorker : public VideoEffect {
    Q_OBJECT
public:
    VHueShiftWorker(float hue, float saturation, float lightness, QObject* parent = nullptr)
        : VideoEffect(parent), m_hue(hue), m_saturation(saturation), m_lightness(lightness) {}
    void process() override;
private:
    float m_hue;
    float m_saturation;
    float m_lightness;
};
