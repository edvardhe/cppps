//
// Created by edvard on 2025-06-09.
//

#include "PhotoDirectoryWidget.h"
#include <QFileDialog>
#include <QImageReader>
#include <QApplication>
#include <QStandardPaths>

PhotoDirectoryWidget::PhotoDirectoryWidget(QWidget *parent)
    : QWidget(parent)
    , m_mainLayout(nullptr)
    , m_pathLayout(nullptr)
    , m_pathLabel(nullptr)
    , m_pathLineEdit(nullptr)
    , m_photoListWidget(nullptr)
{
    // Initialize supported image formats
    m_supportedFormats << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp"
                      << "*.gif" << "*.tiff" << "*.webp";

    setupUI();

    // Set default directory to Pictures folder
    QString picturesPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    if (!picturesPath.isEmpty()) {
        setPhotoDirectory(picturesPath);
    }
}

void PhotoDirectoryWidget::setupUI() {
    // Main layout
    m_mainLayout = new QVBoxLayout(this);

    // Directory path section
    m_pathLayout = new QHBoxLayout();
    m_pathLabel = new QLabel("Photo Directory:");
    m_pathLineEdit = new QLineEdit();
    m_pathLineEdit->setReadOnly(true);
    m_refreshButton = new QPushButton("Refresh");

    m_pathLayout->addWidget(m_pathLabel);
    m_pathLayout->addWidget(m_pathLineEdit, 1); // Stretch factor 1
    m_pathLayout->addWidget(m_refreshButton);

    // Photo list widget
    m_photoListWidget = new QListWidget();
    m_photoListWidget->setViewMode(QListView::IconMode);
    m_photoListWidget->setIconSize(QSize(120, 120));
    m_photoListWidget->setResizeMode(QListView::Adjust);
    m_photoListWidget->setMovement(QListView::Static);

    // Add to main layout
    m_mainLayout->addLayout(m_pathLayout);
    m_mainLayout->addWidget(m_photoListWidget, 1); // Stretch factor 1

    // Connect signals
    connect(m_refreshButton, &QPushButton::clicked,
            this, &PhotoDirectoryWidget::refreshPhotoList);
}

void PhotoDirectoryWidget::setPhotoDirectory(const QString &directoryPath) {
    if (m_currentDirectory != directoryPath) {
        m_currentDirectory = directoryPath;
        m_pathLineEdit->setText(directoryPath);
        refreshPhotoList();
    }
}

QString PhotoDirectoryWidget::getPhotoDirectory() const {
    return m_currentDirectory;
}

void PhotoDirectoryWidget::onDirectoryPathChanged() {
    QString newPath = m_pathLineEdit->text();
    if (newPath != m_currentDirectory) {
        m_currentDirectory = newPath;
        refreshPhotoList();
    }
}

void PhotoDirectoryWidget::refreshPhotoList() {
    m_photoListWidget->clear();

    if (m_currentDirectory.isEmpty()) {
        return;
    }

    QDir directory(m_currentDirectory);
    if (!directory.exists()) {
        return;
    }

    loadPhotosFromDirectory();
}

void PhotoDirectoryWidget::loadPhotosFromDirectory() {
    QDir directory(m_currentDirectory);

    // Set name filters for image files
    directory.setNameFilters(m_supportedFormats);
    directory.setFilter(QDir::Files);

    QFileInfoList fileList = directory.entryInfoList();

    for (const QFileInfo &fileInfo : fileList) {
        if (isImageFile(fileInfo.fileName())) {
            QString filePath = fileInfo.absoluteFilePath();
            QString fileName = fileInfo.baseName();

            // Create thumbnail
            QPixmap thumbnail = createThumbnail(filePath);

            // Create list item
            QListWidgetItem *item = new QListWidgetItem();
            item->setIcon(QIcon(thumbnail));
            item->setText(fileName);
            item->setToolTip(filePath);

            m_photoListWidget->addItem(item);
        }
    }
}

QPixmap PhotoDirectoryWidget::createThumbnail(const QString &imagePath, const QSize &size) {
    QPixmap originalPixmap(imagePath);

    if (originalPixmap.isNull()) {
        // Return a placeholder if image can't be loaded
        QPixmap placeholder(size);
        placeholder.fill(Qt::lightGray);
        return placeholder;
    }

    return originalPixmap.scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

bool PhotoDirectoryWidget::isImageFile(const QString &fileName) {
    QStringList imageExtensions = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"};

    QString extension = QFileInfo(fileName).suffix().toLower();
    return imageExtensions.contains(extension);
}

#include "PhotoDirectoryWidget.moc"