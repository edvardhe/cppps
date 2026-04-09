//
// Created by edvard on 2025-06-09.
//

#include "ObjectParametersWidget.h"
#include <QLabel>
#include <QVBoxLayout>

ObjectParametersWidget::ObjectParametersWidget(QWidget *parent)
    : QWidget(parent)
    , m_initialDepth(nullptr)
    , m_initialAlbedo(nullptr)
{
    setupUI();
}

void ObjectParametersWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);
    auto* formLayout = new QFormLayout();

    // Initial Depth Estimate
    m_initialDepth = new QDoubleSpinBox();
    m_initialDepth->setRange(0.0, 1000.0);
    m_initialDepth->setDecimals(0);
    m_initialDepth->setSuffix(" cm");
    m_initialDepth->setValue(100.0);
    formLayout->addRow("Initial Depth Estimate:", m_initialDepth);

    // Initial Albedo
    m_initialAlbedo = new QDoubleSpinBox();
    m_initialAlbedo->setRange(0.0, 1.0);
    m_initialAlbedo->setDecimals(3);
    m_initialAlbedo->setSingleStep(0.01);
    m_initialAlbedo->setValue(1);
    formLayout->addRow("Initial Albedo:", m_initialAlbedo);

    // Add form layout and stretch
    mainLayout->addLayout(formLayout);
    mainLayout->addStretch();

    // Connect signals
    connect(m_initialDepth, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ObjectParametersWidget::onAnyValueChanged);
    connect(m_initialAlbedo, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ObjectParametersWidget::onAnyValueChanged);
}

ObjectConfig ObjectParametersWidget::config() const {
    ObjectConfig config;
    config.initialDepth = m_initialDepth->value();
    config.initialAlbedo = m_initialAlbedo->value();
    return config;
}

void ObjectParametersWidget::setConfig(const ObjectConfig& config) {
    const QSignalBlocker blockDepth(m_initialDepth);
    const QSignalBlocker blockAlbedo(m_initialAlbedo);

    m_initialDepth->setValue(config.initialDepth);
    m_initialAlbedo->setValue(config.initialAlbedo);
}

void ObjectParametersWidget::onAnyValueChanged() {
    emit objectConfigChanged(config());
}

#include "ObjectParametersWidget.moc"
