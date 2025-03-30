#include "EffectButton.h"

EffectButton::EffectButton(const QString& imagePath, QWidget* parent)
    : QPushButton(parent), pixmap(imagePath)
{
    setStyleSheet("border: none; background: none;");
    setCursor(Qt::PointingHandCursor);

    // Set up zoom animation
    zoomAnimation = new QPropertyAnimation(this, "zoomFactor", this);
    zoomAnimation->setDuration(150); // milliseconds
    zoomAnimation->setEasingCurve(QEasingCurve::OutQuad); // Smooth easing

    // Set up shadow effect
    shadowEffect = new QGraphicsDropShadowEffect(this);
    shadowEffect->setBlurRadius(12);
    shadowEffect->setColor(Qt::black);
    shadowEffect->setOffset(3, 3);
    this->setGraphicsEffect(shadowEffect);

    updateIcon();
}

EffectButton::~EffectButton() {
    delete zoomAnimation;
    delete shadowEffect;
}

void EffectButton::setShadow(qreal blurRadius, const QColor& color, qreal xOffset, qreal yOffset) {
    shadowEffect->setBlurRadius(blurRadius);
    shadowEffect->setColor(color);
    shadowEffect->setOffset(xOffset, yOffset);
}

qreal EffectButton::zoomFactor() const
{
    return m_zoomFactor;
}

void EffectButton::setZoomFactor(qreal factor) {
    if (qFuzzyCompare(m_zoomFactor, factor))
        return;

    m_zoomFactor = factor;
    updateIcon();
}

void EffectButton::resizeEvent(QResizeEvent* event)
{
    QPushButton::resizeEvent(event);
    updateIcon();
}

void EffectButton::enterEvent(QEnterEvent* event)
{
    QPushButton::enterEvent(event);
    zoomAnimation->stop();
    zoomAnimation->setStartValue(m_zoomFactor);
    zoomAnimation->setEndValue(HOVER_ZOOM);
    zoomAnimation->start();

    // Optional: Enhance shadow on hover
    QPropertyAnimation* shadowAnim = new QPropertyAnimation(shadowEffect, "blurRadius", this);
    shadowAnim->setDuration(150);
    shadowAnim->setStartValue(shadowEffect->blurRadius());
    shadowAnim->setEndValue(shadowEffect->blurRadius() * 1.5);
    shadowAnim->start(QAbstractAnimation::DeleteWhenStopped);
}

void EffectButton::leaveEvent(QEvent* event)
{
    QPushButton::leaveEvent(event);
    zoomAnimation->stop();
    zoomAnimation->setStartValue(m_zoomFactor);
    zoomAnimation->setEndValue(1.0);
    zoomAnimation->start();

    // Optional: Return shadow to normal
    QPropertyAnimation* shadowAnim = new QPropertyAnimation(shadowEffect, "blurRadius", this);
    shadowAnim->setDuration(150);
    shadowAnim->setStartValue(shadowEffect->blurRadius());
    shadowAnim->setEndValue(shadowEffect->blurRadius() / 1.5);
    shadowAnim->start(QAbstractAnimation::DeleteWhenStopped);
}

void EffectButton::updateIcon()
{
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

    // Apply zoom factor
    QSize scaledSize = btnSize * m_zoomFactor;
    QPoint offset((btnSize.width() - scaledSize.width()) / 2,
        (btnSize.height() - scaledSize.height()) / 2);

    painter.drawPixmap(offset, pixmap.scaled(scaledSize, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));

    setIcon(QIcon(roundedPixmap));
    setIconSize(btnSize);
}
