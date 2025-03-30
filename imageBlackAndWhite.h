#pragma once

#include "imageEffectBase.h"

class IBlackAndWhiteWorker : public ImageEffect {
    Q_OBJECT
public:
    IBlackAndWhiteWorker(float middle, QObject* parent = nullptr);
    void process() override;
private:
    float m_middle;
};
