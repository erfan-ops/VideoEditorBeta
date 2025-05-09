#include "Blur.h"
#include "imageBlur.h"
#include "image.h"


void IBlurWorker::process() {
    try {
        Image img(m_inputPath);

        BlurProcessor blurProcessor(img.getSize(), img.getWidth(), img.getHeight(), m_blurRadius);
        
        blurProcessor.setImage(img.getData());
        blurProcessor.process();
        memcpy(img.getData(), blurProcessor.getImage(), img.getSize());
        
        img.save(m_outputPath);

        emit finished();
    }
    catch (const std::exception& e) {
        emit errorOccurred(QString("Outline Error: %1").arg(e.what()));
    }
}
