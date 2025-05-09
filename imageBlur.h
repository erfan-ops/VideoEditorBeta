#pragma once

#include "imageEffectBase.h"


class IBlurWorker : public ImageEffect {
    Q_OBJECT
public:
    IBlurWorker(int blurRadius, QObject* parent = nullptr) : ImageEffect(parent), m_blurRadius(blurRadius) {}
    void process() override;
private:
    int m_blurRadius;
};