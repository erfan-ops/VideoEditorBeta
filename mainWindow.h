#pragma once

#include <QMainWindow>
#include <QPushButton>
#include <functional>
#include "effectBase.h" 

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow* ui;
    std::wstring selectedFilePath;
    std::wstring saveFilePath;

    void processEffect(QPushButton* button, EffectBase* worker);
    void replaceButtonWithEffectButton(QPushButton*& button, const QString& imagePath);
};

