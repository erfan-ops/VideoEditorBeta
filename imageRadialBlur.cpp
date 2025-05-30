#include "radialBlur.h"
#include "imageRadialBlur.h"
#include "image.h"


void IRadialBlurWorker::process() {
    try {
        Image img(m_inputPath);

        if (m_centerX < 0.0f || m_centerX >= img.getWidth()) m_centerX = img.getWidth() * 0.5f;
        if (m_centerY < 0.0f || m_centerY >= img.getHeight()) m_centerY = img.getHeight() * 0.5f;

        RadialBlurProcessor processor(img.getSize(), img.getWidth(), img.getHeight(), m_centerX, m_centerY, m_blurRadius, m_intensity);

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
