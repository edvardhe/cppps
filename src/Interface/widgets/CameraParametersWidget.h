//
// Created by edvard on 2025-06-09.
//

#ifndef CAMERAPARAMETERSWIDGET_H
#define CAMERAPARAMETERSWIDGET_H



#include <QWidget>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QFormLayout>
#include "../../ProblemConfig.h"

class CameraParametersWidget : public QWidget {
    Q_OBJECT

public:
    explicit CameraParametersWidget(QWidget *parent = nullptr);

    CameraConfig config() const;
    void setConfig(const CameraConfig& config);
    void updateFromDirectory(const QString& directoryPath);

signals:
    void cameraConfigChanged(const CameraConfig& config);

private slots:
    void onAnyValueChanged();

private:
    void setupUI();

    QDoubleSpinBox* m_focalLengthSpin;
    QDoubleSpinBox* m_sensorWidthSpin;
    QDoubleSpinBox* m_sensorHeightSpin;
    QSpinBox* m_imageWidthSpin;
    QSpinBox* m_imageHeightSpin;
    QDoubleSpinBox* m_principalPointOffsetXSpin;
    QDoubleSpinBox* m_principalPointOffsetYSpin;
};



#endif //CAMERAPARAMETERSWIDGET_H
