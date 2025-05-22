#include "outlines.h"
#include "imageOutlines.h"
#include "image.h"


void IOutlinesWorker::process() {
    try {
        Image img(m_inputPath);
        
        OutlinesProcessor processor(img.getSize(), img.getWidth(), img.getHeight(), m_thicknessX, m_thicknessY);

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
