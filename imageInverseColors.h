#pragma once

#include "imageEffectBase.h"

class IInverseColorsWorker : public ImageEffect {
    Q_OBJECT
public:
    IInverseColorsWorker(QObject* parent = nullptr) {}
    void process() override;
};
