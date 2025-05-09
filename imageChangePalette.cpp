#include "changePalette.h"
#include "imageChangePalette.h"
#include "image.h"


void IChangePaletteWorker::process() {
    try {
        Image img(m_inputPath);

        ChangePaletteProcessor changePaletteProcessor(img.getSize(), img.getNumPixels(), m_colorsBGR, m_numColors);

        changePaletteProcessor.setImage(img.getData());
        changePaletteProcessor.process();
        changePaletteProcessor.upload(img.getData());

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
