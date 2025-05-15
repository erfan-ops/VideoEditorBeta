#include "monoChrome.h"
#include "imageMonoChrome.h"
#include "image.h"


void IMonoChromeWorker::process() {
    try {
        Image img(m_inputPath);

        MonoChromeProcessor processor(img.getSize(), img.getNumPixels());
        
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
