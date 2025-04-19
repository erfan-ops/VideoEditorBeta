#pragma once

#include "imageEffectBase.h"

class ILensFilterWorker : public ImageEffect {
    Q_OBJECT
public:
    ILensFilterWorker(const float* passThreshValues, QObject* parent = nullptr) {
        m_passThreshValues = new float[3];
        std::copy(passThreshValues, passThreshValues + 3, m_passThreshValues);
    }
    void process() override;

private:
    float* m_passThreshValues;
};
