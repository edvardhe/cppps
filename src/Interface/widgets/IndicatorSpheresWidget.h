//
// Created by edvard on 2025-06-09.
//

#ifndef INDICATORSPHERESWIDGET_H
#define INDICATORSPHERESWIDGET_H

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QListWidget>
#include <QPushButton>
#include <QLabel>
#include <QDoubleSpinBox>
#include "../../ProblemConfig.h"

class DraggableSphereMarker : public QGraphicsEllipseItem {
public:
    DraggableSphereMarker(int index, const QString& name, QGraphicsItem* parent = nullptr);

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

class IndicatorSpheresWidget : public QWidget {
    Q_OBJECT

public:
    explicit IndicatorSpheresWidget(QWidget *parent = nullptr);

    IndicatorSphereConfig config() const;
    void setConfig(const IndicatorSphereConfig& config);
    void setPhotoDirectory(const QString& directory);

signals:
    void indicatorConfigChanged(const IndicatorSphereConfig& config);

private slots:
    void onImageSelected(QListWidgetItem* item);
    void onAddSphere();
    void onRemoveSphere();
    void onMarkerMoved();
    void onSphereSelected(int index);
    void onRadiusChanged(double radius);
    void onXChanged(double x);
    void onYChanged(double y);

private:
    void setupUI();
    void loadImageList();
    void loadImage(const QString& imagePath);
    void updateMarkers();
    void updateZoomView();
    void emitConfigChanged();

    QString m_photoDirectory;
    IndicatorSphereConfig m_config;

    // UI Components
    QListWidget* m_imageList;
    QGraphicsView* m_graphicsView;
    QGraphicsScene* m_graphicsScene;
    QGraphicsView* m_zoomView;
    QGraphicsScene* m_zoomScene;
    QPushButton* m_addSphereBtn;
    QPushButton* m_removeSphereBtn;
    QListWidget* m_sphereList;
    QDoubleSpinBox* m_radiusSpin;
    QDoubleSpinBox* m_xSpin;
    QDoubleSpinBox* m_ySpin;

    QGraphicsPixmapItem* m_imageItem;
    QGraphicsPixmapItem* m_zoomImageItem;
    QVector<DraggableSphereMarker*> m_markerItems;
    DraggableSphereMarker* m_selectedMarker;
};

#endif // INDICATORSPHERESWIDGET_H
