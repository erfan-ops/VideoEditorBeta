#include "posterize.h"
#include "imagePosterize.h"
#include "image.h"


void IPosterizeWorker::process() {
    try {
        Image img(m_inputPath);

        unsigned char* d_img;

        PosterizeProcessor processor(img.getSize(), m_threshold);

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
