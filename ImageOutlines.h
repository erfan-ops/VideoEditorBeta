#pragma once

#include "imageEffectBase.h"


class IOutlinesWorker : public ImageEffect {
    Q_OBJECT
public:
    IOutlinesWorker(int thicknessX, int thicknessY, QObject* parent = nullptr);
    void process() override;
private:
    int m_thicknessX;
    int m_thicknessY;
};