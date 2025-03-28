#pragma once

#include "imageEffectBase.h"


class IBlurWorker : public ImageEffect {
    Q_OBJECT
public:
    IBlurWorker(int blurRadius, QObject* parent = nullptr);
    void process() override;
private:
    int m_blurRadius;
};