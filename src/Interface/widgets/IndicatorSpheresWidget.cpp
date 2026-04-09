//
// Created by edvard on 2025-06-09.
//

#include "IndicatorSpheresWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QDir>
#include <QFileInfo>
#include <QGraphicsSceneMouseEvent>
#include <QBrush>
#include <QPen>
#include <QInputDialog>

// DraggableSphereMarker implementation
DraggableSphereMarker::DraggableSphereMarker(int index, const QString& name, QGraphicsItem* parent)
    : QGraphicsEllipseItem(parent), m_index(index)
{
    setFlag(QGraphicsItem::ItemIsMovable);
    setFlag(QGraphicsItem::ItemIsSelectable);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges);

    // Create circle with semi-transparent fill
    setBrush(QBrush(QColor(255, 0, 0, 80)));
    setPen(QPen(QColor(255, 0, 0, 255), 2));
}

void DraggableSphereMarker::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsEllipseItem::mousePressEvent(event);
}

void DraggableSphereMarker::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsEllipseItem::mouseMoveEvent(event);
}

void DraggableSphereMarker::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsEllipseItem::mouseReleaseEvent(event);
    // Notify parent widget that marker was moved
    if (scene()) {
        // The widget will connect to scene's changed signal
    }
}

// IndicatorSpheresWidget implementation
IndicatorSpheresWidget::IndicatorSpheresWidget(QWidget *parent)
    : QWidget(parent)
    , m_imageItem(nullptr)
    , m_zoomImageItem(nullptr)
    , m_selectedMarker(nullptr)
{
    setupUI();
}

void IndicatorSpheresWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // Main content: splitter with 3 panels
    auto* splitter = new QSplitter(Qt::Horizontal);

    // Left panel: Image list
    auto* imageListGroup = new QGroupBox("Reference Images");
    auto* imageListLayout = new QVBoxLayout(imageListGroup);
    m_imageList = new QListWidget();
    imageListLayout->addWidget(m_imageList);
    splitter->addWidget(imageListGroup);

    // Center panel: Graphics view
    auto* viewGroup = new QGroupBox("Image Viewer");
    auto* viewLayout = new QVBoxLayout(viewGroup);
    m_graphicsScene = new QGraphicsScene(this);
    m_graphicsView = new QGraphicsView(m_graphicsScene);
    m_graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    m_graphicsView->setRenderHint(QPainter::Antialiasing);
    viewLayout->addWidget(m_graphicsView);
    splitter->addWidget(viewGroup);

    // Right panel: Split vertically for sphere list and zoom view
    auto* rightWidget = new QWidget();
    auto* rightLayout = new QVBoxLayout(rightWidget);
    rightLayout->setContentsMargins(0, 0, 0, 0);

    // Top half: Sphere list and controls
    auto* controlGroup = new QGroupBox("Sphere Markers");
    auto* controlLayout = new QVBoxLayout(controlGroup);

    m_sphereList = new QListWidget();
    controlLayout->addWidget(m_sphereList);

    m_addSphereBtn = new QPushButton("Add Sphere");
    m_removeSphereBtn = new QPushButton("Remove Selected");
    controlLayout->addWidget(m_addSphereBtn);
    controlLayout->addWidget(m_removeSphereBtn);

    // Radius control
    auto* radiusLayout = new QHBoxLayout();
    radiusLayout->addWidget(new QLabel("Radius:"));
    m_radiusSpin = new QDoubleSpinBox();
    m_radiusSpin->setRange(5.0, 500.0);
    m_radiusSpin->setValue(20.0);
    m_radiusSpin->setSuffix(" px");
    radiusLayout->addWidget(m_radiusSpin);
    controlLayout->addLayout(radiusLayout);

    // X position control
    auto* xLayout = new QHBoxLayout();
    xLayout->addWidget(new QLabel("X:"));
    m_xSpin = new QDoubleSpinBox();
    m_xSpin->setRange(0.0, 100000.0);
    m_xSpin->setValue(0.0);
    m_xSpin->setDecimals(1);
    m_xSpin->setSuffix(" px");
    xLayout->addWidget(m_xSpin);
    controlLayout->addLayout(xLayout);

    // Y position control
    auto* yLayout = new QHBoxLayout();
    yLayout->addWidget(new QLabel("Y:"));
    m_ySpin = new QDoubleSpinBox();
    m_ySpin->setRange(0.0, 100000.0);
    m_ySpin->setValue(0.0);
    m_ySpin->setDecimals(1);
    m_ySpin->setSuffix(" px");
    yLayout->addWidget(m_ySpin);
    controlLayout->addLayout(yLayout);


    rightLayout->addWidget(controlGroup);

    // Bottom half: Zoom view
    auto* zoomGroup = new QGroupBox("Zoom View");
    auto* zoomLayout = new QVBoxLayout(zoomGroup);
    m_zoomScene = new QGraphicsScene(this);
    m_zoomView = new QGraphicsView(m_zoomScene);
    m_zoomView->setRenderHint(QPainter::Antialiasing);
    m_zoomView->setMinimumHeight(150);
    zoomLayout->addWidget(m_zoomView);
    rightLayout->addWidget(zoomGroup);

    splitter->addWidget(rightWidget);

    // Set splitter proportions (1:4:1.5)
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 4);
    splitter->setStretchFactor(2, 1);

    mainLayout->addWidget(splitter);

    // Connect signals
    connect(m_imageList, &QListWidget::itemClicked, this, &IndicatorSpheresWidget::onImageSelected);
    connect(m_addSphereBtn, &QPushButton::clicked, this, &IndicatorSpheresWidget::onAddSphere);
    connect(m_removeSphereBtn, &QPushButton::clicked, this, &IndicatorSpheresWidget::onRemoveSphere);
    connect(m_graphicsScene, &QGraphicsScene::changed, this, &IndicatorSpheresWidget::onMarkerMoved);
    connect(m_sphereList, &QListWidget::currentRowChanged, this, &IndicatorSpheresWidget::onSphereSelected);
    connect(m_radiusSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &IndicatorSpheresWidget::onRadiusChanged);
    connect(m_xSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &IndicatorSpheresWidget::onXChanged);
    connect(m_ySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &IndicatorSpheresWidget::onYChanged);
}

void IndicatorSpheresWidget::setPhotoDirectory(const QString& directory) {
    m_photoDirectory = directory;
    loadImageList();
}

void IndicatorSpheresWidget::loadImageList() {
    m_imageList->clear();

    if (m_photoDirectory.isEmpty()) {
        return;
    }

    QDir dir(m_photoDirectory);
    if (!dir.exists()) {
        return;
    }

    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.tiff" << "*.webp";
    QFileInfoList fileList = dir.entryInfoList(filters, QDir::Files);

    for (const QFileInfo& fileInfo : fileList) {
        m_imageList->addItem(fileInfo.fileName());
    }
}

void IndicatorSpheresWidget::onImageSelected(QListWidgetItem* item) {
    if (!item || m_photoDirectory.isEmpty()) {
        return;
    }

    QString imagePath = QDir(m_photoDirectory).filePath(item->text());
    loadImage(imagePath);
    m_config.referenceImagePath = imagePath;
    emitConfigChanged();
}

void IndicatorSpheresWidget::loadImage(const QString& imagePath) {
    QPixmap pixmap(imagePath);
    if (pixmap.isNull()) {
        return;
    }

    m_graphicsScene->clear();
    m_markerItems.clear();

    m_imageItem = m_graphicsScene->addPixmap(pixmap);
    m_graphicsScene->setSceneRect(pixmap.rect());

    // Restore existing markers
    updateMarkers();

    // Fit in view
    m_graphicsView->fitInView(m_graphicsScene->sceneRect(), Qt::KeepAspectRatio);
}

void IndicatorSpheresWidget::onAddSphere() {
    bool ok;
    QString name = QInputDialog::getText(this, "Add Sphere",
                                         "Enter sphere name:",
                                         QLineEdit::Normal,
                                         QString("Sphere %1").arg(m_config.sphereMarkers.size() + 1),
                                         &ok);
    if (!ok || name.isEmpty()) {
        return;
    }

    SphereMarker marker;
    marker.name = name;
    marker.x = m_graphicsScene->sceneRect().center().x();
    marker.y = m_graphicsScene->sceneRect().center().y();
    marker.radius = 20.0;

    m_config.sphereMarkers.append(marker);
    updateMarkers();
    emitConfigChanged();
}

void IndicatorSpheresWidget::onRemoveSphere() {
    int selectedRow = m_sphereList->currentRow();
    if (selectedRow < 0 || selectedRow >= m_config.sphereMarkers.size()) {
        return;
    }

    m_config.sphereMarkers.removeAt(selectedRow);
    updateMarkers();
    emitConfigChanged();
}

void IndicatorSpheresWidget::updateMarkers() {
    // Clear existing marker items
    for (auto* marker : m_markerItems) {
        m_graphicsScene->removeItem(marker);
        delete marker;
    }
    m_markerItems.clear();

    // Update sphere list
    int previousSelection = m_sphereList->currentRow();
    m_sphereList->clear();

    // Create new marker items
    for (int i = 0; i < m_config.sphereMarkers.size(); ++i) {
        const SphereMarker& marker = m_config.sphereMarkers[i];

        auto* item = new DraggableSphereMarker(i, marker.name);
        double r = marker.radius;
        item->setRect(-r, -r, 2*r, 2*r);
        item->setPos(marker.x, marker.y);
        m_graphicsScene->addItem(item);
        m_markerItems.append(item);

        m_sphereList->addItem(marker.name);
    }

    // Restore selection or select first item
    if (m_config.sphereMarkers.size() > 0) {
        if (previousSelection >= 0 && previousSelection < m_config.sphereMarkers.size()) {
            m_sphereList->setCurrentRow(previousSelection);
        } else {
            m_sphereList->setCurrentRow(0);
        }
    } else {
        m_selectedMarker = nullptr;
        updateZoomView();
    }
}

void IndicatorSpheresWidget::onMarkerMoved() {
    // Update config from marker positions
    for (int i = 0; i < m_markerItems.size() && i < m_config.sphereMarkers.size(); ++i) {
        QPointF pos = m_markerItems[i]->pos();
        m_config.sphereMarkers[i].x = pos.x();
        m_config.sphereMarkers[i].y = pos.y();
    }

    // Update X and Y spinboxes for selected marker
    int selectedRow = m_sphereList->currentRow();
    if (selectedRow >= 0 && selectedRow < m_config.sphereMarkers.size()) {
        m_xSpin->blockSignals(true);
        m_xSpin->setValue(m_config.sphereMarkers[selectedRow].x);
        m_xSpin->blockSignals(false);

        m_ySpin->blockSignals(true);
        m_ySpin->setValue(m_config.sphereMarkers[selectedRow].y);
        m_ySpin->blockSignals(false);
    }

    updateZoomView();
    emitConfigChanged();
}

void IndicatorSpheresWidget::onSphereSelected(int index) {
    if (index < 0 || index >= m_markerItems.size()) {
        m_selectedMarker = nullptr;
        return;
    }

    m_selectedMarker = m_markerItems[index];

    // Update radius, X, and Y spinboxes
    if (index < m_config.sphereMarkers.size()) {
        m_radiusSpin->blockSignals(true);
        m_radiusSpin->setValue(m_config.sphereMarkers[index].radius);
        m_radiusSpin->blockSignals(false);

        m_xSpin->blockSignals(true);
        m_xSpin->setValue(m_config.sphereMarkers[index].x);
        m_xSpin->blockSignals(false);

        m_ySpin->blockSignals(true);
        m_ySpin->setValue(m_config.sphereMarkers[index].y);
        m_ySpin->blockSignals(false);
    }

    updateZoomView();
}

void IndicatorSpheresWidget::onRadiusChanged(double radius) {
    int selectedRow = m_sphereList->currentRow();
    if (selectedRow < 0 || selectedRow >= m_config.sphereMarkers.size()) {
        return;
    }

    m_config.sphereMarkers[selectedRow].radius = radius;

    // Update the visual marker
    if (selectedRow < m_markerItems.size()) {
        double r = radius;
        m_markerItems[selectedRow]->setRect(-r, -r, 2*r, 2*r);
    }

    updateZoomView();
    emitConfigChanged();
}

void IndicatorSpheresWidget::updateZoomView() {
    m_zoomScene->clear();
    m_zoomImageItem = nullptr;

    if (!m_selectedMarker || !m_imageItem) {
        return;
    }

    // Get the selected marker's position
    QPointF markerPos = m_selectedMarker->pos();
    double radius = m_selectedMarker->rect().width() / 2.0;

    // Define zoom region (3x the marker radius)
    double zoomRadius = radius * 3.0;
    QRectF zoomRect(markerPos.x() - zoomRadius,
                    markerPos.y() - zoomRadius,
                    zoomRadius * 2,
                    zoomRadius * 2);

    // Ensure zoom rect is within image bounds
    QRectF imageRect = m_imageItem->boundingRect();
    zoomRect = zoomRect.intersected(imageRect);

    // Copy the zoomed portion of the image
    QPixmap fullPixmap = m_imageItem->pixmap();
    QPixmap zoomedPixmap = fullPixmap.copy(zoomRect.toRect());

    m_zoomImageItem = m_zoomScene->addPixmap(zoomedPixmap);
    m_zoomScene->setSceneRect(zoomedPixmap.rect());

    // Draw the marker circle on the zoom view
    QPointF relativePos = markerPos - zoomRect.topLeft();
    auto* zoomMarker = m_zoomScene->addEllipse(
        relativePos.x() - radius,
        relativePos.y() - radius,
        radius * 2,
        radius * 2,
        QPen(QColor(255, 0, 0, 255), 2),
        QBrush(QColor(255, 0, 0, 80))
    );

    m_zoomView->fitInView(m_zoomScene->sceneRect(), Qt::KeepAspectRatio);
}

void IndicatorSpheresWidget::onXChanged(double x) {
    int selectedRow = m_sphereList->currentRow();
    if (selectedRow < 0 || selectedRow >= m_config.sphereMarkers.size()) {
        return;
    }

    m_config.sphereMarkers[selectedRow].x = x;

    // Update the visual marker position
    if (selectedRow < m_markerItems.size()) {
        m_markerItems[selectedRow]->setPos(x, m_config.sphereMarkers[selectedRow].y);
    }

    updateZoomView();
    emitConfigChanged();
}

void IndicatorSpheresWidget::onYChanged(double y) {
    int selectedRow = m_sphereList->currentRow();
    if (selectedRow < 0 || selectedRow >= m_config.sphereMarkers.size()) {
        return;
    }

    m_config.sphereMarkers[selectedRow].y = y;

    // Update the visual marker position
    if (selectedRow < m_markerItems.size()) {
        m_markerItems[selectedRow]->setPos(m_config.sphereMarkers[selectedRow].x, y);
    }

    updateZoomView();
    emitConfigChanged();
}

void IndicatorSpheresWidget::emitConfigChanged() {
    emit indicatorConfigChanged(m_config);
}

IndicatorSphereConfig IndicatorSpheresWidget::config() const {
    return m_config;
}

void IndicatorSpheresWidget::setConfig(const IndicatorSphereConfig& config) {
    m_config = config;

    // Load the reference image if available
    if (!m_config.referenceImagePath.isEmpty() && QFileInfo::exists(m_config.referenceImagePath)) {
        loadImage(m_config.referenceImagePath);

        // Select the image in the list
        QString fileName = QFileInfo(m_config.referenceImagePath).fileName();
        auto items = m_imageList->findItems(fileName, Qt::MatchExactly);
        if (!items.isEmpty()) {
            m_imageList->setCurrentItem(items.first());
        }
    }

    updateMarkers();
}

#include "IndicatorSpheresWidget.moc"
