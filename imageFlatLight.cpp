#include "flatLight.h"
#include "imageFlatLight.h"
#include "image.h"


void IFlatLightWorker::process() {
    try {
        Image img(m_inputPath);

        FlatLightProcessor flatLightProcessor(img.getSize(), img.getNumPixels(), m_lightness);

        flatLightProcessor.setImage(img.getData());
        flatLightProcessor.process();
        flatLightProcessor.upload(img.getData());

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
