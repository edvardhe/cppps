//
// Created by edvard on 2025-06-09.
//

#include "CameraParametersWidget.h"
#include <QLabel>
#include <QDir>
#include <QFileInfo>
#include <QImageReader>

CameraParametersWidget::CameraParametersWidget(QWidget *parent)
    : QWidget(parent)
    , m_focalLengthSpin(nullptr)
    , m_sensorWidthSpin(nullptr)
    , m_sensorHeightSpin(nullptr)
    , m_imageWidthSpin(nullptr)
    , m_imageHeightSpin(nullptr)
    , m_principalPointOffsetXSpin(nullptr)
    , m_principalPointOffsetYSpin(nullptr)
{
    setupUI();
}

void CameraParametersWidget::setupUI() {
    auto* layout = new QFormLayout(this);

    // Focal Length
    m_focalLengthSpin = new QDoubleSpinBox();
    m_focalLengthSpin->setRange(0.0, 10000.0);
    m_focalLengthSpin->setDecimals(2);
    m_focalLengthSpin->setSuffix(" mm");
    layout->addRow("Focal Length:", m_focalLengthSpin);

    // Sensor Width
    m_sensorWidthSpin = new QDoubleSpinBox();
    m_sensorWidthSpin->setRange(0.0, 1000.0);
    m_sensorWidthSpin->setDecimals(2);
    m_sensorWidthSpin->setSuffix(" mm");
    layout->addRow("Sensor Width:", m_sensorWidthSpin);

    // Sensor Height
    m_sensorHeightSpin = new QDoubleSpinBox();
    m_sensorHeightSpin->setRange(0.0, 1000.0);
    m_sensorHeightSpin->setDecimals(2);
    m_sensorHeightSpin->setSuffix(" mm");
    layout->addRow("Sensor Height:", m_sensorHeightSpin);

    // Image Width
    m_imageWidthSpin = new QSpinBox();
    m_imageWidthSpin->setRange(0, 100000);
    m_imageWidthSpin->setSuffix(" px");
    m_imageWidthSpin->setEnabled(false); // Auto-fetched from image
    layout->addRow("Image Width:", m_imageWidthSpin);

    // Image Height
    m_imageHeightSpin = new QSpinBox();
    m_imageHeightSpin->setRange(0, 100000);
    m_imageHeightSpin->setSuffix(" px");
    m_imageHeightSpin->setEnabled(false); // Auto-fetched from image
    layout->addRow("Image Height:", m_imageHeightSpin);

    // Principal Point Offset X
    m_principalPointOffsetXSpin = new QDoubleSpinBox();
    m_principalPointOffsetXSpin->setRange(-10000.0, 10000.0);
    m_principalPointOffsetXSpin->setDecimals(2);
    m_principalPointOffsetXSpin->setSuffix(" px");
    m_principalPointOffsetXSpin->setValue(0.0);
    layout->addRow("Principal Point Offset X:", m_principalPointOffsetXSpin);

    // Principal Point Offset Y
    m_principalPointOffsetYSpin = new QDoubleSpinBox();
    m_principalPointOffsetYSpin->setRange(-10000.0, 10000.0);
    m_principalPointOffsetYSpin->setDecimals(2);
    m_principalPointOffsetYSpin->setSuffix(" px");
    m_principalPointOffsetYSpin->setValue(0.0);
    layout->addRow("Principal Point Offset Y:", m_principalPointOffsetYSpin);

    // Connect signals
    connect(m_focalLengthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &CameraParametersWidget::onAnyValueChanged);
    connect(m_sensorWidthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &CameraParametersWidget::onAnyValueChanged);
    connect(m_sensorHeightSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &CameraParametersWidget::onAnyValueChanged);
    connect(m_imageWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &CameraParametersWidget::onAnyValueChanged);
    connect(m_imageHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &CameraParametersWidget::onAnyValueChanged);
    connect(m_principalPointOffsetXSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &CameraParametersWidget::onAnyValueChanged);
    connect(m_principalPointOffsetYSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &CameraParametersWidget::onAnyValueChanged);
}

CameraConfig CameraParametersWidget::config() const {
    CameraConfig config;
    config.focalLengthMm = m_focalLengthSpin->value();
    config.sensorWidthMm = m_sensorWidthSpin->value();
    config.sensorHeightMm = m_sensorHeightSpin->value();
    config.imageWidthPx = m_imageWidthSpin->value();
    config.imageHeightPx = m_imageHeightSpin->value();
    config.principalPointOffsetX = m_principalPointOffsetXSpin->value();
    config.principalPointOffsetY = m_principalPointOffsetYSpin->value();
    return config;
}

void CameraParametersWidget::setConfig(const CameraConfig& config) {
    m_focalLengthSpin->setValue(config.focalLengthMm);
    m_sensorWidthSpin->setValue(config.sensorWidthMm);
    m_sensorHeightSpin->setValue(config.sensorHeightMm);
    m_imageWidthSpin->setValue(config.imageWidthPx);
    m_imageHeightSpin->setValue(config.imageHeightPx);
    m_principalPointOffsetXSpin->setValue(config.principalPointOffsetX);
    m_principalPointOffsetYSpin->setValue(config.principalPointOffsetY);
}

void CameraParametersWidget::updateFromDirectory(const QString& directoryPath) {
    if (directoryPath.isEmpty()) {
        return;
    }

    QDir directory(directoryPath);
    if (!directory.exists()) {
        return;
    }

    // Look for image files
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.tiff" << "*.webp";
    directory.setNameFilters(filters);
    directory.setFilter(QDir::Files);

    QFileInfoList fileList = directory.entryInfoList();
    if (fileList.isEmpty()) {
        return;
    }

    // Use the first image to get dimensions
    QString firstImagePath = fileList.first().absoluteFilePath();

    QImageReader reader(firstImagePath);
    QSize imageSize = reader.size();

    if (imageSize.isValid()) {
        m_imageWidthSpin->setValue(imageSize.width());
        m_imageHeightSpin->setValue(imageSize.height());

        // Try to read physical DPI to calculate sensor size
        // Default to 72 DPI if not available
        QImage image = reader.read();
        if (!image.isNull()) {
            int dpiX = image.dotsPerMeterX() > 0 ? image.dotsPerMeterX() : 2835; // 72 DPI in dots per meter
            int dpiY = image.dotsPerMeterY() > 0 ? image.dotsPerMeterY() : 2835;

            // Calculate sensor size from image dimensions and DPI
            // Convert from dots per meter to dots per mm: dots/meter * 0.001 = dots/mm
            double dotsPerMmX = dpiX * 0.001;
            double dotsPerMmY = dpiY * 0.001;

            if (dotsPerMmX > 0 && dotsPerMmY > 0) {
                double sensorWidthMm = imageSize.width() / dotsPerMmX;
                double sensorHeightMm = imageSize.height() / dotsPerMmY;

                m_sensorWidthSpin->setValue(sensorWidthMm);
                m_sensorHeightSpin->setValue(sensorHeightMm);
            }
        }
    }
}

void CameraParametersWidget::onAnyValueChanged() {
    emit cameraConfigChanged(config());
}

#include "CameraParametersWidget.moc"
