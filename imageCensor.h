#pragma once

#include "imageEffectBase.h"

class ICensorWorker : public ImageEffect {
    Q_OBJECT
public:
    ICensorWorker(int pixelWidth, int pixelHeight, QObject* parent = nullptr) : ImageEffect(parent), m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight) {};
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
};
