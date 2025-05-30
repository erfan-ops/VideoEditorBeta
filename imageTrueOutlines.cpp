#include "trueOutlines.h"
#include "imageTrueOutlines.h"
#include "image.h"


void ITrueOutlinesWorker::process() {
    try {
        Image img(m_inputPath);
        
        TrueOutlinesProcessor processor(img.getSize(), img.getWidth(), img.getHeight(), m_thresh);

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
