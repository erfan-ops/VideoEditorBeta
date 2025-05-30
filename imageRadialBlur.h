#pragma once

#include "imageEffectBase.h"


class IRadialBlurWorker : public ImageEffect {
    Q_OBJECT
public:
    IRadialBlurWorker(int blurRadius, float intensity, float centerX, float centerY , QObject* parent = nullptr)
        : ImageEffect(parent), m_blurRadius(blurRadius), m_intensity(intensity), m_centerX(centerX), m_centerY(centerY) {}

    void process() override;
private:
    int m_blurRadius;
    float m_intensity;
    float m_centerX;
    float m_centerY;
};
