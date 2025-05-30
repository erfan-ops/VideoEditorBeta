#include "vintage8Bit.h"
#include "imageVintage8bit.h"
#include "image.h"


void IVintage8bitWorker::process() {
    try {
        Image img(m_inputPath);

        Vintage8BitProcessor processor(img.getSize(), img.getWidth(), img.getHeight(), m_pixelWidth, m_pixelHeight, m_thresh);

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
