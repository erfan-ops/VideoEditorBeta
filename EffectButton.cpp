#include "EffectButton.h"


EffectButton::EffectButton(const QString& imagePath, QWidget* parent)
    : QPushButton(parent), pixmap(imagePath)
{
    setStyleSheet("border: none; background: none;");
    setCursor(Qt::PointingHandCursor);
    updateIcon();
}

void EffectButton::resizeEvent(QResizeEvent* event)
{
    QPushButton::resizeEvent(event);
    updateIcon();
}

void EffectButton::updateIcon() {
    if (pixmap.isNull()) return;

    QSize btnSize = size();
    QPixmap roundedPixmap(btnSize);
    roundedPixmap.fill(Qt::transparent);

    QPainter painter(&roundedPixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    QPainterPath path;
    int radius = 16;
    path.addRoundedRect(roundedPixmap.rect(), radius, radius);

    painter.setClipPath(path);
    painter.drawPixmap(0, 0, pixmap.scaled(btnSize, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));

    setIcon(QIcon(roundedPixmap));
    setIconSize(btnSize);
}
