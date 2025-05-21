#include "pixelate.h"
#include "imagePixelate.h"
#include "image.h"


void IPixelateWorker::process() {
    try {
        Image img(m_inputPath);
        
        PixelateProcessor processor(img.getSize(), img.getWidth(), img.getHeight(), m_pixelWidth, m_pixelHeight);

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
