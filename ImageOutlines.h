#pragma once

#include "imageEffectBase.h"


class IOutlinesWorker : public ImageEffect {
    Q_OBJECT
public:
    IOutlinesWorker(int thicknessX, int thicknessY, QObject* parent = nullptr)
        : ImageEffect(parent), m_thicknessX(thicknessX), m_thicknessY(thicknessY) {}
    void process() override;
private:
    int m_thicknessX;
    int m_thicknessY;
};