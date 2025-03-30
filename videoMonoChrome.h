#pragma once

#include "videoEffectBase.h"

class VMonoChromeWorker : public VideoEffect {
    Q_OBJECT
public:
    VMonoChromeWorker(QObject* parent = nullptr) {};
    void process() override;
};
