#include "ReviewAndSelectWidget.h"

ReviewAndSelectWidget::ReviewAndSelectWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

void ReviewAndSelectWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // Camera properties display (read-only, few lines)
    m_cameraPropertiesDisplay = new QTextEdit(this);
    m_cameraPropertiesDisplay->setReadOnly(true);
    m_cameraPropertiesDisplay->setMaximumHeight(100);  // Limit height to a few lines
    m_cameraPropertiesDisplay->setStyleSheet("QTextEdit { background-color: #f5f5f5; border: 1px solid #ccc; }");
    mainLayout->addWidget(m_cameraPropertiesDisplay);

    // Instruction label
    m_selectInstructionLabel = new QLabel("Select region of interest for optimization", this);
    m_selectInstructionLabel->setStyleSheet("QLabel { font-weight: bold; margin-top: 8px; margin-bottom: 4px; }");
    mainLayout->addWidget(m_selectInstructionLabel);

    // ROI list
    m_roiList = new QListWidget(this);
    m_roiList->setSelectionMode(QAbstractItemView::SingleSelection);
    mainLayout->addWidget(m_roiList);

    connect(m_roiList, &QListWidget::currentRowChanged, this, [this](int) {
        onSelectionChanged();
    });
}

void ReviewAndSelectWidget::setCameraProperties(const CameraConfig& camera) {
    updateCameraParametersDisplay(camera);
}

void ReviewAndSelectWidget::updateCameraParametersDisplay(const CameraConfig& camera) {
    QString text = QString(
        "<b>Camera Properties:</b><br>"
        "Focal Length: %1 mm | "
        "Sensor: %2 × %3 mm | "
        "Image: %4 × %5 px | "
        "Principal Point Offset: (%6, %7)"
    )
        .arg(camera.focalLengthMm, 0, 'f', 2)
        .arg(camera.sensorWidthMm, 0, 'f', 2)
        .arg(camera.sensorHeightMm, 0, 'f', 2)
        .arg(camera.imageWidthPx)
        .arg(camera.imageHeightPx)
        .arg(camera.principalPointOffsetX, 0, 'f', 2)
        .arg(camera.principalPointOffsetY, 0, 'f', 2);
    
    m_cameraPropertiesDisplay->setHtml(text);
}

void ReviewAndSelectWidget::setRois(const QVector<ROIConfig>& rois){
    m_rois = rois;
    m_roiList->clear();

    for (const ROIConfig& roi : rois){
        m_roiList->addItem(roi.name);
    }
}

int ReviewAndSelectWidget::selectedRoiIndex() const {
    return m_roiList->currentRow();
}

void ReviewAndSelectWidget::setSelectedRoiIndex(int index) {
    if (index >= 0 && index < m_roiList->count()) {
        m_roiList->setCurrentRow(index);
    }
}

void ReviewAndSelectWidget::onSelectionChanged() {
    emit selectedRoiChanged();
}