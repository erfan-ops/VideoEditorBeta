#pragma once

#include "imageEffectBase.h"

class IFlatLightWorker : public ImageEffect {
    Q_OBJECT
public:
    IFlatLightWorker(float lightness, QObject* parent = nullptr) : ImageEffect(parent), m_lightness(lightness) {}
    void process() override;
private:
    float m_lightness;
};
