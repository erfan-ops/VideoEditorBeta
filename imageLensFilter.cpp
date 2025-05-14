#include "lensFilter.h"
#include "imageLensFilter.h"
#include "image.h"


void ILensFilterWorker::process() {
    try {
        Image img(m_inputPath);

        LensFilterProcessor processor(img.getSize(), m_passThreshValues, 3);

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
