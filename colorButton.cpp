#include "colorButton.h"

#include <QString>

ColorButton::ColorButton(QColor color, int index, QWidget* parent)
    : QPushButton(parent), color(color), index(index)
{
    setFixedSize(100, 30);
    setColor(color);
}

void ColorButton::setColor(const QColor& color) {
    this->color = color;

    // Maintain consistent styling (background, border, border-radius)
    QString style = QString(
        "background-color: %1;"
        "border: 2px solid %2;"
        "border-radius: 4px;"
    )
        .arg(color.name())  // e.g., #ff0000
        .arg(isSelected ? "#0078d7" : "transparent");  // Use selection logic

    this->setStyleSheet(style);
}


QColor ColorButton::getColor() const {
    return color;
}
