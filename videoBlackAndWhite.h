#pragma once

#include "videoEffectBase.h"


class VBlackAndWhiteWorker : public VideoEffect {
    Q_OBJECT
public:
    VBlackAndWhiteWorker(float middle, QObject* parent = nullptr) : VideoEffect(parent), m_middle(middle) {}
    void process() override;
private:
    float m_middle;
};
