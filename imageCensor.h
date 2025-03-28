#pragma once

#include "imageEffectBase.h"

class ICensorWorker : public ImageEffect {
    Q_OBJECT
public:
    ICensorWorker(int pixelWidht, int pixelHeight, QObject* parent = nullptr);
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
};
