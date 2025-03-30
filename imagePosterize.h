#pragma once

#include "imageEffectBase.h"

class IPosterizeWorker : public ImageEffect {
    Q_OBJECT
public:
    IPosterizeWorker(int threshold, QObject* parent = nullptr);
    void process() override;
private:
    int m_threshold;
};
