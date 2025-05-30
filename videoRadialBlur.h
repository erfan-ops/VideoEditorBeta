#pragma once

#include "videoEffectBase.h"


class VRadialBlurWorker : public VideoEffect {
    Q_OBJECT
public:
    VRadialBlurWorker(int blurRadius, float intensity, float centerX, float centerY, QObject* parent = nullptr)
        : VideoEffect(parent), m_blurRadius(blurRadius), m_intensity(intensity), m_centerX(centerX), m_centerY(centerY) {}

    void process() override;
private:
    int m_blurRadius;
    float m_intensity;
    float m_centerX;
    float m_centerY;
};
