#pragma once

#include "videoEffectBase.h"


class VVintage8bitWorker : public VideoEffect {
    Q_OBJECT
public:
    VVintage8bitWorker(int pixelWidth, int pixelHeight, int thresh, QObject* parent = nullptr)
        : m_thresh(thresh), m_pixelWidth(pixelWidth), m_pixelHeight(pixelHeight) {};
    void process() override;
private:
    int m_pixelWidth;
    int m_pixelHeight;
    int m_thresh;
};
