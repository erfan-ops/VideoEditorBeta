#include "mainWindow.h"
#include <QApplication>
#include <QMetaType>

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);
    MainWindow w(argc, argv);
    w.show();
    return a.exec();
}
