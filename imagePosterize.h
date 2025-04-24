#pragma once

#include "imageEffectBase.h"

class IPosterizeWorker : public ImageEffect {
    Q_OBJECT
public:
    IPosterizeWorker(float threshold, QObject* parent = nullptr) : m_threshold(255.0f / threshold) {}
    void process() override;
private:
    float m_threshold;
};
