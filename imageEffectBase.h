#pragma once

#include <effectBase.h>
#include <QObject>


class ImageEffect : public EffectBase {
    Q_OBJECT
public:
    explicit ImageEffect(QObject* parent = nullptr) : EffectBase(parent) {}
    virtual ~ImageEffect() = default;

    virtual void process() override = 0;
};
