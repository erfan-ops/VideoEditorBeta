#pragma once

#include "videoEffectBase.h"

class VPosterizeWorker : public VideoEffect {
    Q_OBJECT
public:
    VPosterizeWorker(int threshold, QObject* parent = nullptr) : m_threshold(threshold) {};
    void process() override;
private:
    int m_threshold;
};
