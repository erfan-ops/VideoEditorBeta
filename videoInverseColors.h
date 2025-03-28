#pragma once

#include "videoEffectBase.h"

class VInverseColorsWorker : public VideoEffect {
    Q_OBJECT
public:
    VInverseColorsWorker(QObject* parent = nullptr) {};
    void process() override;
};
