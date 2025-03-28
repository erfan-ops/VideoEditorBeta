#pragma once

#include "videoEffectBase.h"


class VOutlineWorker : public VideoEffect {
    Q_OBJECT
public:
    VOutlineWorker(int shiftX, int shiftY, QObject* parent = nullptr);
    void process() override;
private:
    int m_shiftX, m_shiftY;
};