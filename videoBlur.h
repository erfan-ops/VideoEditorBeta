#pragma once

#include "videoEffectBase.h"


class VBlurWorker : public VideoEffect {
    Q_OBJECT
public:
    VBlurWorker(int blurRadius, QObject* parent = nullptr);
    void process() override;
private:
    int m_blurRadius;
};
