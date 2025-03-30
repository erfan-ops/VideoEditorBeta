#pragma once

#include "imageEffectBase.h"

class IMonoChromeWorker : public ImageEffect {
    Q_OBJECT
public:
    IMonoChromeWorker(QObject* parent = nullptr);
    void process() override;
};
