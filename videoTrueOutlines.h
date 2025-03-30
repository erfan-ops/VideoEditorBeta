#pragma once

#include "videoEffectBase.h"


class VTrueOutlinesWorker : public VideoEffect {
    Q_OBJECT
public:
    VTrueOutlinesWorker(int thresh, QObject* parent = nullptr) : m_thresh(thresh) {};
    void process() override;
private:
    int m_thresh;
};
