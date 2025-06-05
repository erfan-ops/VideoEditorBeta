#pragma once

#include "imageEffectBase.h"

class IHueShiftWorker : public ImageEffect {
    Q_OBJECT
public:
    IHueShiftWorker(float hue, float saturation, float lightness, QObject* parent = nullptr)
        : ImageEffect(parent), m_hue(hue), m_saturation(saturation), m_lightness(lightness) {}
    void process() override;
private:
    float m_hue;
    float m_saturation;
    float m_lightness;
};
