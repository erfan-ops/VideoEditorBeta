#include "imageSoftPalette.h"
#include "image.h"
#include "softPalette.h"


void ISoftPaletteWorker::process() {
    try {
        Image img(m_inputPath);

        SoftPaletteProcessor softPaletteProcessor(img.getNumPixels(), img.getSize(), m_colorsBGR, m_numColors);

        softPaletteProcessor.upload(img.getData());
        softPaletteProcessor.process();
        softPaletteProcessor.download(img.getData());

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
