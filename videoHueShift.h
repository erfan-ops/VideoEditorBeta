#pragma once

#include "videoEffectBase.h"


class VHueShiftWorker : public VideoEffect {
    Q_OBJECT
public:
    VHueShiftWorker(float shift, QObject* parent = nullptr)
        : VideoEffect(parent), m_shift(shift / 180.0f) {}
    void process() override;
private:
    float m_shift;
};
