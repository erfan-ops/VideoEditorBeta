#include "inverseContrast.h"
#include "imageInverseContrast.h"
#include "image.h"


void IInverseContrastWorker::process() {
    try {
        Image img(m_inputPath);

        InverseContrastProcessor processor(img.getSize(), img.getNumPixels());

        processor.setImage(img.getData());
        processor.process();
        processor.upload(img.getData());

        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
