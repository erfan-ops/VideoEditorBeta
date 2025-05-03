#pragma once

#include <QPushButton>
#include <QWidget>
#include <QColor>

class ColorButton : public QPushButton {
    Q_OBJECT

public:
    explicit ColorButton(QColor color, int index = -1, QWidget* parent = nullptr);

    void setIndex(int idx) { index = idx; }
    int getIndex() const { return index; }

    void setColor(const QColor& color);
    QColor getColor() const;

    void setSelected(bool selected) { isSelected = selected; }

private:
    int index;
    QColor color;

    bool isSelected = false;
};