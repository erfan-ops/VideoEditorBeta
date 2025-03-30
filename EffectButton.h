#pragma once

#include <QPushButton>
#include <QPixmap>
#include <QPainter>
#include <QPainterPath>
#include <QWidget>
#include <QString>
#include <QPropertyAnimation>
#include <QGraphicsDropShadowEffect>

class EffectButton : public QPushButton
{
    Q_OBJECT
        Q_PROPERTY(qreal zoomFactor READ zoomFactor WRITE setZoomFactor)

public:
    explicit EffectButton(const QString& imagePath, QWidget* parent = nullptr);
    ~EffectButton();

    qreal zoomFactor() const;
    void setZoomFactor(qreal factor);

    void setShadow(qreal blurRadius = 10.0, const QColor& color = Qt::black,
        qreal xOffset = 0.0, qreal yOffset = 2.0);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    QPixmap pixmap;
    QPropertyAnimation* zoomAnimation;
    QGraphicsDropShadowEffect* shadowEffect;
    qreal m_zoomFactor = 1.0;
    const qreal HOVER_ZOOM = 1.2; // 20% zoom on hover

    void updateIcon();
};
