#pragma once

#include "videoEffectBase.h"

class VInverseContrastWorker : public VideoEffect {
    Q_OBJECT
public:
    VInverseContrastWorker(QObject* parent = nullptr) {};
    void process() override;
};
