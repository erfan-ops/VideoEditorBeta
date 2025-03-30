#pragma once

#include "imageEffectBase.h"

class ITrueOutlinesWorker : public ImageEffect {
    Q_OBJECT
public:
    ITrueOutlinesWorker(int thresh, QObject* parent = nullptr) : m_thresh(thresh) {};
    void process() override;
private:
    float m_thresh;
};
