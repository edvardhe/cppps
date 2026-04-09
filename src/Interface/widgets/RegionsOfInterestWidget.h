//
// Created by edvard on 2025-06-09.
//

#ifndef REGIONSOFINTERESTWIDGET_H
#define REGIONSOFINTERESTWIDGET_H

#include <QWidget>
#include <QListWidget>
#include <QSpinBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include "../../ProblemConfig.h"

class DraggableROIMarker : public QGraphicsRectItem {
public:
    DraggableROIMarker(int index, const QString& name, QGraphicsItem* parent = nullptr);

    int index() const { return m_index; }
    void setIndex(int index) { m_index = index; }

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    int m_index;
    QGraphicsTextItem* m_label;
};

class RegionsOfInterestWidget : public QWidget {
    Q_OBJECT

public:
    explicit RegionsOfInterestWidget(QWidget *parent = nullptr);

    QVector<ROIConfig> rois() const;
    void setRois(const QVector<ROIConfig>& rois);
    void setPhotoDirectory(const QString& directory);

signals:
    void roisChanged(const QVector<ROIConfig>& rois);
    void selectedRoiChanged(int index);

private slots:
    void onSelectionChanged();
    void onAddRoi();
    void onRemoveRoi();
    void onImageSelected(QListWidgetItem* item);
    void onMarkerMoved();
    void onRoiSelected(int index);
    void onParameterChanged();

private:
    void setupUI();
    void loadImageList();
    void loadImage(const QString& imagePath);
    void updateMarkers();
    void updateZoomedView(int roiIndex);
    void emitConfigChanged();

    QString m_photoDirectory;
    QVector<ROIConfig> m_rois;

    // UI Components
    QListWidget* m_imageList;
    QGraphicsView* m_graphicsView;
    QGraphicsScene* m_graphicsScene;
    QListWidget* m_roiList;
    QGraphicsView* m_zoomedGraphicsView;
    QGraphicsScene* m_zoomedGraphicsScene;
    QSpinBox* m_startX;
    QSpinBox* m_startY;
    QSpinBox* m_roiWidth;
    QSpinBox* m_roiHeight;
    QSpinBox* m_gridX;
    QSpinBox* m_gridY;
    QPushButton* m_addButton;
    QPushButton* m_removeButton;

    QGraphicsPixmapItem* m_imageItem;
    QVector<DraggableROIMarker*> m_markerItems;
    DraggableROIMarker* m_selectedMarker;
};

#endif // REGIONSOFINTERESTWIDGET_H
