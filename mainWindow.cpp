#include "mainWindow.h"
#include "ui_mainwindow.h"

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QMessageBox>
#include <QProgressBar>
#include <QThread>
#include <QPixmap>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>

#include <QDebug>

#include "filedialog.h"
#include "effects.h"
#include "utils.h"

#include "EffectButton.h"


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/icon.ico"));

    // Connect signals
    QObject::connect(ui->btnSelect, &QPushButton::clicked, [&]() {
        selectedFilePath = FileDialog::OpenFileDialog(L"All Files");
        if (!selectedFilePath.empty()) {
            ui->label->setText(QString::fromStdWString(L"Selected: " + selectedFilePath));
        }
        });

    replaceButtonWithEffectButton(ui->btnBlur, ":/samples/samples/blur.jpg");
    replaceButtonWithEffectButton(ui->btnOutlines, ":/samples/samples/outline.jpg");
    replaceButtonWithEffectButton(ui->btnPixelate, ":/samples/samples/pixelate.jpg");
    replaceButtonWithEffectButton(ui->btnCensor, ":/samples/samples/pixelate.jpg");

    QObject::connect(ui->btnBlur, &QPushButton::clicked, this, [&]() {
        // Get effect parameters
        int blurRadius = ui->blurRadius->value();

        // Create the appropriate worker
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VBlurWorker(blurRadius);
            QObject::connect(worker, &EffectBase::progressChanged, ui->progressBar, &QProgressBar::setValue, Qt::QueuedConnection);
        }
        else {
            worker = new IBlurWorker(blurRadius);
        }

        // Start processing
        processEffect(ui->btnBlur, worker);
        });

    QObject::connect(ui->btnOutlines, &QPushButton::clicked, this, [&]() {
        // Get effect parameters
        const int thicknessX = ui->ThicknessX->value();
        const int thicknessY = ui->ThicknessY->value();

        // Create the appropriate worker
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VOutlineWorker(thicknessX, thicknessY);
            QObject::connect(worker, &EffectBase::progressChanged, ui->progressBar, &QProgressBar::setValue, Qt::QueuedConnection);
        }
        else {
            worker = new IOutlinesWorker(thicknessX, thicknessY);
        }

        // Start processing
        processEffect(ui->btnBlur, worker);
        });

    QObject::connect(ui->btnPixelate, &QPushButton::clicked, this, [&]() {
        // Get effect parameters
        const int pixelWidth = ui->pixelWidth->value();
        const int pixelHeight = ui->pixelHeight->value();

        // Create the appropriate worker
        EffectBase* worker = nullptr;
        if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
            worker = new VPixelateWorker(pixelWidth, pixelHeight);
            QObject::connect(worker, &EffectBase::progressChanged, ui->progressBar, &QProgressBar::setValue, Qt::QueuedConnection);
        }
        else {
            worker = new IPixelateWorker(pixelWidth, pixelHeight);
        }

        // Start processing
        processEffect(ui->btnBlur, worker);
        });
}

MainWindow::~MainWindow()
{
    delete ui;
}


void  MainWindow::processEffect(QPushButton* button, EffectBase* worker) {
    if (selectedFilePath.empty()) {
        QMessageBox::warning(this, "Error", "Please select a file first!");
        return;
    }

    if (videoExtentions.find(fileUtils::splitextw(selectedFilePath).second) != videoExtentions.end()) {
        saveFilePath = FileDialog::SaveFileDialog(L"Video Files", L"output.mp4");
    }
    else {
        saveFilePath = FileDialog::SaveFileDialog(L"Image Files", L"output.png");
    }
    if (saveFilePath.empty()) return;

    button->setEnabled(false);

    QThread* workerThread = new QThread();
    worker->setInputPath(selectedFilePath);
    worker->setOutputPath(saveFilePath);
    worker->moveToThread(workerThread);

    QObject::connect(workerThread, &QThread::started, worker, &EffectBase::process);
    QObject::connect(worker, &EffectBase::finished, workerThread, &QThread::quit);
    QObject::connect(worker, &EffectBase::errorOccurred, workerThread, &QThread::quit);

    QObject::connect(worker, &EffectBase::finished, this, [this, btn = QPointer<QPushButton>(button)]() {
        btn->setEnabled(true);
        QMessageBox::information(this, "Success", "Image saved at \"" + QString::fromStdWString(saveFilePath) + '"');
        });

    QObject::connect(workerThread, &QThread::finished, worker, &EffectBase::deleteLater);
    QObject::connect(workerThread, &QThread::finished, workerThread, &QThread::deleteLater);

    workerThread->start();
}


void MainWindow::replaceButtonWithEffectButton(QPushButton*& button, const QString& imagePath)
{
    if (!button) return; // Safety check

    EffectButton* effectBtn = new EffectButton(imagePath, this);
    effectBtn->setGeometry(button->geometry());
    effectBtn->show();

    delete button;  // Remove the original button
    button = effectBtn; // Update pointer to the new button
}
