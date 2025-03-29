#pragma once

#include "imageEffectBase.h"

class IHueShiftWorker : public ImageEffect {
    Q_OBJECT
public:
    IHueShiftWorker(float shift, QObject* parent = nullptr);
    void process() override;
private:
    float m_shift;
};
