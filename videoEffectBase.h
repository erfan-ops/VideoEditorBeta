#pragma once

#include <QObject>
#include <string>

class VideoEffect : public QObject {
    Q_OBJECT
public:
    explicit VideoEffect(QObject* parent = nullptr) : QObject(parent) {}
    virtual ~VideoEffect() = default;

    virtual void process() = 0;

    void setInputPath(const std::wstring& path) { m_inputPath = path; }
    void setOutputPath(const std::wstring& path) { m_outputPath = path; }

signals:
    void progressChanged(int percent);
    void finished();
    void errorOccurred(const QString& message);

public:
    std::wstring m_inputPath;
    std::wstring m_outputPath;
};
