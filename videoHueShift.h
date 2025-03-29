#pragma once

#include "videoEffectBase.h"


class VHueShiftWorker : public VideoEffect {
    Q_OBJECT
public:
    VHueShiftWorker(float shift, QObject* parent = nullptr);
    void process() override;
private:
    float m_shift;
};
