#pragma once

#include "imageEffectBase.h"

class IHueShiftWorker : public ImageEffect {
    Q_OBJECT
public:
    IHueShiftWorker(float shift, QObject* parent = nullptr)
        : ImageEffect(parent), m_shift(shift / 180.0f) {}
    void process() override;
private:
    float m_shift;
};
