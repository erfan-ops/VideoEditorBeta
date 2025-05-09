#include "blackAndWhite.h"
#include "imageBlackAndWhite.h"
#include "image.h"


void IBlackAndWhiteWorker::process() {
    try {
        Image img(m_inputPath);

        BlackAndWhiteProcessor blackAndWhiteProcessor(img.getNumPixels(), img.getSize(), m_middle);

        blackAndWhiteProcessor.setImage(img.getData(), img.getSize());
        blackAndWhiteProcessor.process();
        memcpy(img.getData(), blackAndWhiteProcessor.getImage(), img.getSize());
        
        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
