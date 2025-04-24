#pragma once

#include "videoEffectBase.h"

class VPosterizeWorker : public VideoEffect {
    Q_OBJECT
public:
    VPosterizeWorker(float threshold, QObject* parent = nullptr) : m_threshold(255.0f / threshold) {};
    void process() override;
private:
    float m_threshold;
};
