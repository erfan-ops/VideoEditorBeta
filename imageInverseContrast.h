#pragma once

#include "imageEffectBase.h"

class IInverseContrastWorker : public ImageEffect {
    Q_OBJECT
public:
    IInverseContrastWorker(QObject* parent = nullptr);
    void process() override;
};
