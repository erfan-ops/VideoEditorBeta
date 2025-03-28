#pragma once

#include <QObject>
#include "effectBase.h"


class VideoEffect : public EffectBase {
    Q_OBJECT
public:
    explicit VideoEffect(QObject* parent = nullptr) : EffectBase(parent) {}
    virtual ~VideoEffect() = default;

    virtual void process() override = 0;
};
