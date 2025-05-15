#include "monoMask.h"
#include "imageMonoMask.h"
#include "image.h"


void IMonoMaskWorker::process() {
    try {
        Image img(m_inputPath);

        MonoMaskProcessor processor(img.getNumPixels(), img.getSize(), m_colorsBGR, m_numColors);

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
