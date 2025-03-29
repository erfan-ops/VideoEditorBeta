#pragma once

#include "imageEffectBase.h"


class IRadialBlurWorker : public ImageEffect {
    Q_OBJECT
public:
    IRadialBlurWorker(int blurRadius, float intensity, float centerX, float centerY , QObject* parent = nullptr);
    void process() override;
private:
    int m_blurRadius;
    float m_intensity;
    float m_centerX;
    float m_centerY;
};
