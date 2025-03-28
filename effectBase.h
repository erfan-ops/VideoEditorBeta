#pragma once

#include <QObject>

class EffectBase : public QObject {
    Q_OBJECT
public:
    explicit EffectBase(QObject* parent = nullptr) : QObject(parent) {}
    virtual ~EffectBase() = default;

    virtual void process() = 0;

    void setInputPath(const std::wstring& path) { m_inputPath = path; }
    void setOutputPath(const std::wstring& path) { m_outputPath = path; }

signals:
    void progressChanged(int percent);
    void finished();
    void errorOccurred(const QString& message);

protected:
    std::wstring m_inputPath;
    std::wstring m_outputPath;
};