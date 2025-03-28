#pragma once

#include <QObject>
#include "effectBase.h"


class ImageEffect : public EffectBase {
    Q_OBJECT
public:
    explicit ImageEffect(QObject* parent = nullptr) : EffectBase(parent) {}
    virtual ~ImageEffect() = default;

    virtual void process() override = 0;
};
