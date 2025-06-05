#include "flatSaturation.h"
#include "imageFlatSaturation.h"
#include "image.h"


void IFlatSaturationWorker::process() {
    try {
        Image img(m_inputPath);

        FlatSaturationProcessor processor(img.getSize(), img.getNumPixels(), m_saturation);

        processor.upload(img.getData());
        processor.process();
        processor.download(img.getData());

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
