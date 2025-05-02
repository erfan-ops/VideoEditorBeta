#pragma once

#include "imageEffectBase.h"

class IChangePaletteWorker : public ImageEffect {
    Q_OBJECT
public:
    IChangePaletteWorker(unsigned char* colorsBGR, int numColors, QObject* parent = nullptr)
        : m_colorsBGR(colorsBGR), m_numColors(numColors) {};
    void process() override;
private:
    unsigned char* m_colorsBGR;
    int m_numColors;
};
