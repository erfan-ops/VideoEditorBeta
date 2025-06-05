#pragma once

#include "imageEffectBase.h"

class IFlatSaturationWorker : public ImageEffect {
    Q_OBJECT
public:
    IFlatSaturationWorker(float saturation, QObject* parent = nullptr) : ImageEffect(parent), m_saturation(saturation) {}
    void process() override;
private:
    float m_saturation;
};
