#pragma once

#include <QPushButton>
#include <QPixmap>
#include <QPainter>
#include <QPainterPath>
#include <QWidget>
#include <QString>

class EffectButton : public QPushButton
{
    Q_OBJECT
public:
    explicit EffectButton(const QString& imagePath, QWidget* parent = nullptr);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    QPixmap pixmap;

    void updateIcon();
};
