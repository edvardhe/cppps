#ifndef REVIEWANDSELECTWIDGET_H
#define REVIEWANDSELECTWIDGET_H

#include <QWidget>
#include <QLabel>
#include <QTextEdit>
#include <QListWidget>
#include <QVBoxLayout>
#include "../../ProblemConfig.h"

class ReviewAndSelectWidget : public QWidget {
    Q_OBJECT
    
public:
    ReviewAndSelectWidget(QWidget *parent = nullptr);

    void setRois(const QVector<ROIConfig>& rois);
    void setCameraProperties(const CameraConfig& camera);

    int selectedRoiIndex() const;
    void setSelectedRoiIndex(int index);

signals:
    void selectedRoiChanged();

private slots:
    void onSelectionChanged();

private:
    void setupUI();
    void updateCameraParametersDisplay(const CameraConfig& camera);

    QVector<ROIConfig> m_rois;

    // UI Components
    QTextEdit* m_cameraPropertiesDisplay;
    QLabel* m_selectInstructionLabel;
    QListWidget* m_roiList;
};
#endif 