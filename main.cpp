﻿#include "mainWindow.h"
#include <QApplication>
#include <QMetaType>

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
