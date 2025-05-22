#pragma once

#include "videoEffectBase.h"


class VOutlineWorker : public VideoEffect {
    Q_OBJECT
public:
    VOutlineWorker(int shiftX, int shiftY, QObject* parent = nullptr)
        : VideoEffect(parent), m_shiftX(shiftX), m_shiftY(shiftY) {}
    void process() override;
private:
    int m_shiftX, m_shiftY;
};