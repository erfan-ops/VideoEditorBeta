#pragma once

#include "videoEffectBase.h"


class VMagicEyeWorker : public VideoEffect {
    Q_OBJECT
public:
    VMagicEyeWorker(float middle, QObject* parent = nullptr) : m_middle(middle) {};
    void process() override;
private:
    float m_middle;
};
