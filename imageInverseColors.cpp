#include "inverseColors.h"
#include "imageInverseColors.h"
#include "image.h"


void IInverseColorsWorker::process() {
    try {
        Image img(m_inputPath);
        
        InverseColorsProcessor processor(img.getSize());

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
