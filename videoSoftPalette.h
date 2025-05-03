#pragma once

#include "videoEffectBase.h"

class VSoftPaletteWorker : public VideoEffect {
    Q_OBJECT
public:
    VSoftPaletteWorker(unsigned char* colorsBGR, int numColors, QObject* parent = nullptr)
        : m_colorsBGR(colorsBGR), m_numColors(numColors) {
    };
    void process() override;
private:
    unsigned char* m_colorsBGR;
    int m_numColors;
};
