#include "hueShift.h"
#include "imageHueShift.h"
#include "image.h"


void IHueShiftWorker::process() {
    try {
        Image img(m_inputPath);
        
        HueShiftProcessor processor(img.getSize(), img.getNumPixels(), m_shift);

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
