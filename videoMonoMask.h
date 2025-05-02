#pragma once

#include "videoEffectBase.h"

class VMonoMaskWorker : public VideoEffect {
    Q_OBJECT
public:
    VMonoMaskWorker(unsigned char* colorsBGR, int numColors, QObject* parent = nullptr)
        : m_colorsBGR(colorsBGR), m_numColors(numColors) {
    };
    void process() override;
private:
    unsigned char* m_colorsBGR;
    int m_numColors;
};
