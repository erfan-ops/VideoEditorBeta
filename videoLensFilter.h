#pragma once

#include "videoEffectBase.h"

class VLensFilterWorker : public VideoEffect {
    Q_OBJECT
public:
    VLensFilterWorker(float* passThreshValues,  QObject* parent = nullptr) {
        m_passThreshValues = new float[3];
        std::copy(passThreshValues, passThreshValues + 3, m_passThreshValues);
    };
    void process() override;

private:
    float* m_passThreshValues;
};
