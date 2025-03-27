#pragma once

#include "videoEffectBase.h"


class OutlineWorker : public VideoEffect {
    Q_OBJECT
public:
    OutlineWorker(int shiftX, int shiftY, QObject* parent = nullptr);
    void process() override;
private:
    int m_shiftX, m_shiftY;
};