#pragma once

#include "videoEffectBase.h"


class VRadialBlurWorker : public VideoEffect {
    Q_OBJECT
public:
    VRadialBlurWorker(int blurRadius, float intensity, float centerX, float centerY, QObject* parent = nullptr);
    void process() override;
private:
    int m_blurRadius;
    float m_intensity;
    float m_centerX;
    float m_centerY;
};
