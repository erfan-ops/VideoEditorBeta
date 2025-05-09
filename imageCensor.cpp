#include "censor.h"
#include "imageCensor.h"
#include "image.h"


void ICensorWorker::process() {
    try {
        Image img(m_inputPath);

        CensorProcessor censorProcessor(img.getSize(), img.getWidth(), img.getHeight(), m_pixelWidth, m_pixelHeight);

        censorProcessor.setImage(img.getData());
        censorProcessor.process();
        censorProcessor.upload(img.getData());

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
