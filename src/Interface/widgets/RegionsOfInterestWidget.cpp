//
// Created by edvard on 2025-06-09.
//

#include "RegionsOfInterestWidget.h"
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QFormLayout>
#include <QDir>
#include <QFileInfo>
#include <QGraphicsSceneMouseEvent>
#include <QBrush>
#include <QPen>
#include <QInputDialog>

// DraggableROIMarker implementation
DraggableROIMarker::DraggableROIMarker(int index, const QString& name, QGraphicsItem* parent)
    : QGraphicsRectItem(parent), m_index(index)
{
    setFlag(QGraphicsItem::ItemIsMovable);
    setFlag(QGraphicsItem::ItemIsSelectable);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges);

    // Create rectangle with semi-transparent fill
    setBrush(QBrush(QColor(0, 255, 0, 50)));
    setPen(QPen(QColor(0, 255, 0, 255), 2));

    // Add label
    m_label = new QGraphicsTextItem(name, this);
    m_label->setDefaultTextColor(Qt::green);
    m_label->setFlag(QGraphicsItem::ItemIgnoresTransformations);
}

void DraggableROIMarker::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsRectItem::mousePressEvent(event);
}

void DraggableROIMarker::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsRectItem::mouseMoveEvent(event);
}

void DraggableROIMarker::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsRectItem::mouseReleaseEvent(event);
}

// RegionsOfInterestWidget implementation
RegionsOfInterestWidget::RegionsOfInterestWidget(QWidget *parent)
    : QWidget(parent)
    , m_imageItem(nullptr)
    , m_selectedMarker(nullptr)
{
    setupUI();
}

void RegionsOfInterestWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // Main content: splitter with 2 panels
    auto* splitter = new QSplitter(Qt::Horizontal);

    // Left panel: Stacked image list and ROI list
    auto* leftWidget = new QWidget();
    auto* leftLayout = new QVBoxLayout(leftWidget);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    // Image list at the top
    auto* imageListGroup = new QGroupBox("Reference Images");
    auto* imageListLayout = new QVBoxLayout(imageListGroup);
    m_imageList = new QListWidget();
    m_imageList->setMaximumHeight(200);
    imageListLayout->addWidget(m_imageList);
    leftLayout->addWidget(imageListGroup);

    // ROI list below
    auto* roiListGroup = new QGroupBox("ROI List");
    auto* roiListLayout = new QVBoxLayout(roiListGroup);
    m_roiList = new QListWidget();
    roiListLayout->addWidget(m_roiList);
    leftLayout->addWidget(roiListGroup);

    splitter->addWidget(leftWidget);

    // Right panel: Main image view + zoomed view side by side, and controls below
    auto* rightWidget = new QWidget();
    auto* rightLayout = new QVBoxLayout(rightWidget);
    rightLayout->setContentsMargins(0, 0, 0, 0);

    // Image views side by side
    auto* viewsLayout = new QHBoxLayout();

    // Main graphics view
    auto* viewGroup = new QGroupBox("Image Viewer");
    auto* viewLayout = new QVBoxLayout(viewGroup);
    m_graphicsScene = new QGraphicsScene(this);
    m_graphicsView = new QGraphicsView(m_graphicsScene);
    m_graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    m_graphicsView->setRenderHint(QPainter::Antialiasing);
    viewLayout->addWidget(m_graphicsView);
    viewsLayout->addWidget(viewGroup);

    // Zoomed graphics view
    auto* zoomedViewGroup = new QGroupBox("Zoomed ROI View");
    auto* zoomedViewLayout = new QVBoxLayout(zoomedViewGroup);
    m_zoomedGraphicsScene = new QGraphicsScene(this);
    m_zoomedGraphicsView = new QGraphicsView(m_zoomedGraphicsScene);
    m_zoomedGraphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
    m_zoomedGraphicsView->setRenderHint(QPainter::Antialiasing);
    zoomedViewLayout->addWidget(m_zoomedGraphicsView);
    viewsLayout->addWidget(zoomedViewGroup);

    rightLayout->addLayout(viewsLayout);

    // ROI Controls
    auto* controlGroup = new QGroupBox("ROI Controls");
    auto* controlLayout = new QVBoxLayout(controlGroup);

    // Buttons
    m_addButton = new QPushButton("Add ROI");
    m_removeButton = new QPushButton("Remove Selected");
    controlLayout->addWidget(m_addButton);
    controlLayout->addWidget(m_removeButton);

    // ROI Parameters
    auto* paramGroup = new QGroupBox("ROI Parameters");
    auto* formLayout = new QFormLayout(paramGroup);

    m_startX = new QSpinBox();
    m_startX->setRange(0, 100000);
    m_startX->setValue(0);
    formLayout->addRow("Start X:", m_startX);

    m_startY = new QSpinBox();
    m_startY->setRange(0, 100000);
    m_startY->setValue(0);
    formLayout->addRow("Start Y:", m_startY);

    m_roiWidth = new QSpinBox();
    m_roiWidth->setRange(1, 100000);
    m_roiWidth->setValue(100);
    formLayout->addRow("Width:", m_roiWidth);

    m_roiHeight = new QSpinBox();
    m_roiHeight->setRange(1, 100000);
    m_roiHeight->setValue(100);
    formLayout->addRow("Height:", m_roiHeight);

    m_gridX = new QSpinBox();
    m_gridX->setRange(1, 1000);
    m_gridX->setValue(10);
    formLayout->addRow("Grid X:", m_gridX);

    m_gridY = new QSpinBox();
    m_gridY->setRange(1, 1000);
    m_gridY->setValue(10);
    formLayout->addRow("Grid Y:", m_gridY);

    controlLayout->addWidget(paramGroup);
    rightLayout->addWidget(controlGroup);
    splitter->addWidget(rightWidget);

    splitter->setSizes({300, 700});
    mainLayout->addWidget(splitter);

    // Connect signals
    connect(m_imageList, &QListWidget::itemClicked, this, &RegionsOfInterestWidget::onImageSelected);
    connect(m_roiList, &QListWidget::currentRowChanged, this, &RegionsOfInterestWidget::onRoiSelected);
    connect(m_addButton, &QPushButton::clicked, this, &RegionsOfInterestWidget::onAddRoi);
    connect(m_removeButton, &QPushButton::clicked, this, &RegionsOfInterestWidget::onRemoveRoi);
    connect(m_graphicsScene, &QGraphicsScene::changed, this, &RegionsOfInterestWidget::onMarkerMoved);

    connect(m_startX, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
    connect(m_startY, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
    connect(m_roiWidth, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
    connect(m_roiHeight, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
    connect(m_gridX, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
    connect(m_gridY, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
}

QVector<ROIConfig> RegionsOfInterestWidget::rois() const {
    return m_rois;
}

void RegionsOfInterestWidget::setRois(const QVector<ROIConfig>& rois) {
    m_rois = rois;
    m_roiList->clear();
    for (const auto& roi : m_rois) {
        m_roiList->addItem(roi.name);
    }
    updateMarkers();
}


void RegionsOfInterestWidget::setPhotoDirectory(const QString& directory) {
    m_photoDirectory = directory;
    loadImageList();
}

void RegionsOfInterestWidget::onSelectionChanged() {
    emit selectedRoiChanged(m_roiList->currentRow());
}

void RegionsOfInterestWidget::onAddRoi() {
    bool ok;
    QString name = QInputDialog::getText(this, "Add ROI",
                                        "ROI name:",
                                        QLineEdit::Normal,
                                        QString("ROI %1").arg(m_rois.size() + 1),
                                        &ok);
    if (!ok || name.isEmpty()) {
        return;
    }

    ROIConfig roi;
    roi.name = name;
    roi.startX = m_startX->value();
    roi.startY = m_startY->value();
    roi.width = m_roiWidth->value();
    roi.height = m_roiHeight->value();
    roi.gridX = m_gridX->value();
    roi.gridY = m_gridY->value();

    m_rois.append(roi);
    m_roiList->addItem(roi.name);
    updateMarkers();
    emitConfigChanged();
}

void RegionsOfInterestWidget::onRemoveRoi() {
    int index = m_roiList->currentRow();
    if (index >= 0 && index < m_rois.size()) {
        m_rois.removeAt(index);
        delete m_roiList->takeItem(index);
        updateMarkers();
        emitConfigChanged();
    }
}

void RegionsOfInterestWidget::onImageSelected(QListWidgetItem* item) {
    if (!item || m_photoDirectory.isEmpty()) {
        return;
    }

    QString imagePath = QDir(m_photoDirectory).filePath(item->text());
    loadImage(imagePath);
}

void RegionsOfInterestWidget::onMarkerMoved() {
    // Update ROI configs from marker positions
    for (int i = 0; i < m_markerItems.size() && i < m_rois.size(); ++i) {
        QRectF rect = m_markerItems[i]->rect();
        QPointF pos = m_markerItems[i]->pos();

        m_rois[i].startX = static_cast<int>(pos.x());
        m_rois[i].startY = static_cast<int>(pos.y());
        m_rois[i].width = static_cast<int>(rect.width());
        m_rois[i].height = static_cast<int>(rect.height());
    }

    // Update spin boxes if a ROI is selected
    int currentRow = m_roiList->currentRow();
    if (currentRow >= 0 && currentRow < m_rois.size()) {
        disconnect(m_startX, nullptr, this, nullptr);
        disconnect(m_startY, nullptr, this, nullptr);
        disconnect(m_roiWidth, nullptr, this, nullptr);
        disconnect(m_roiHeight, nullptr, this, nullptr);

        m_startX->setValue(m_rois[currentRow].startX);
        m_startY->setValue(m_rois[currentRow].startY);
        m_roiWidth->setValue(m_rois[currentRow].width);
        m_roiHeight->setValue(m_rois[currentRow].height);

        connect(m_startX, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_startY, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_roiWidth, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_roiHeight, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);

        // Update zoomed view for the selected ROI
        updateZoomedView(currentRow);
    }

    emitConfigChanged();
}

void RegionsOfInterestWidget::onRoiSelected(int index) {
    if (index >= 0 && index < m_rois.size()) {
        // Update parameter fields
        disconnect(m_startX, nullptr, this, nullptr);
        disconnect(m_startY, nullptr, this, nullptr);
        disconnect(m_roiWidth, nullptr, this, nullptr);
        disconnect(m_roiHeight, nullptr, this, nullptr);
        disconnect(m_gridX, nullptr, this, nullptr);
        disconnect(m_gridY, nullptr, this, nullptr);

        m_startX->setValue(m_rois[index].startX);
        m_startY->setValue(m_rois[index].startY);
        m_roiWidth->setValue(m_rois[index].width);
        m_roiHeight->setValue(m_rois[index].height);
        m_gridX->setValue(m_rois[index].gridX);
        m_gridY->setValue(m_rois[index].gridY);

        connect(m_startX, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_startY, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_roiWidth, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_roiHeight, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_gridX, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);
        connect(m_gridY, QOverload<int>::of(&QSpinBox::valueChanged), this, &RegionsOfInterestWidget::onParameterChanged);

        // Highlight selected marker
        if (index < m_markerItems.size()) {
            m_selectedMarker = m_markerItems[index];
            m_selectedMarker->setSelected(true);
        }

        // Update zoomed view
        updateZoomedView(index);
    }

    emit selectedRoiChanged(index);
}

void RegionsOfInterestWidget::onParameterChanged() {
    int index = m_roiList->currentRow();
    if (index >= 0 && index < m_rois.size()) {
        m_rois[index].startX = m_startX->value();
        m_rois[index].startY = m_startY->value();
        m_rois[index].width = m_roiWidth->value();
        m_rois[index].height = m_roiHeight->value();
        m_rois[index].gridX = m_gridX->value();
        m_rois[index].gridY = m_gridY->value();

        updateMarkers();
        emitConfigChanged();
    }
}

void RegionsOfInterestWidget::loadImageList() {
    m_imageList->clear();

    if (m_photoDirectory.isEmpty()) {
        return;
    }

    QDir dir(m_photoDirectory);
    if (!dir.exists()) {
        return;
    }

    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.tif" << "*.tiff";
    QStringList imageFiles = dir.entryList(filters, QDir::Files, QDir::Name);

    for (const QString& filename : imageFiles) {
        m_imageList->addItem(filename);
    }
}

void RegionsOfInterestWidget::loadImage(const QString& imagePath) {
    QPixmap pixmap(imagePath);
    if (pixmap.isNull()) {
        return;
    }

    m_graphicsScene->clear();
    m_zoomedGraphicsScene->clear();
    m_markerItems.clear();
    m_imageItem = nullptr;

    m_imageItem = m_graphicsScene->addPixmap(pixmap);
    m_graphicsScene->setSceneRect(pixmap.rect());

    updateMarkers();

    m_graphicsView->fitInView(m_imageItem, Qt::KeepAspectRatio);
}

void RegionsOfInterestWidget::updateMarkers() {
    // Clear existing markers
    for (auto* marker : m_markerItems) {
        m_graphicsScene->removeItem(marker);
        delete marker;
    }
    m_markerItems.clear();
    m_selectedMarker = nullptr;

    // Create new markers
    for (int i = 0; i < m_rois.size(); ++i) {
        const auto& roi = m_rois[i];
        auto* marker = new DraggableROIMarker(i, roi.name);
        marker->setRect(0, 0, roi.width, roi.height);
        marker->setPos(roi.startX, roi.startY);

        m_graphicsScene->addItem(marker);
        m_markerItems.append(marker);
    }
}

void RegionsOfInterestWidget::updateZoomedView(int roiIndex) {
    m_zoomedGraphicsScene->clear();

    if (roiIndex < 0 || roiIndex >= m_rois.size() || !m_imageItem) {
        return;
    }

    const auto& roi = m_rois[roiIndex];

    // Create a cropped view of the ROI from the main image
    QPixmap mainPixmap = m_imageItem->pixmap();
    if (mainPixmap.isNull()) {
        return;
    }

    // Extract the ROI region from the main pixmap
    QRect roiRect(roi.startX, roi.startY, roi.width, roi.height);
    QPixmap croppedPixmap = mainPixmap.copy(roiRect);

    if (croppedPixmap.isNull()) {
        return;
    }

    // Add the cropped image to the zoomed scene
    auto* zoomedImageItem = m_zoomedGraphicsScene->addPixmap(croppedPixmap);

    // Add a semi-transparent overlay rectangle to show the grid
    auto* overlayRect = new QGraphicsRectItem(0, 0, roi.width, roi.height);
    overlayRect->setBrush(QBrush(QColor(0, 255, 0, 30)));
    overlayRect->setPen(QPen(QColor(0, 255, 0, 255), 2));
    m_zoomedGraphicsScene->addItem(overlayRect);

    m_zoomedGraphicsScene->setSceneRect(croppedPixmap.rect());

    // Scale to fit the view while maintaining aspect ratio
    m_zoomedGraphicsView->fitInView(zoomedImageItem, Qt::KeepAspectRatio);
}

void RegionsOfInterestWidget::emitConfigChanged() {
    emit roisChanged(m_rois);
}
