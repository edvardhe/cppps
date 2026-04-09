#include "MainWindow.h"
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QMessageBox>
#include <QListWidget>
#include <QLabel>
#include <QFileDialog>
#include <QInputDialog>
#include <QDir>
#include <QJsonObject>
#include <QJsonDocument>
#include <QStandardPaths>
#include <QFile>
#include <QImageReader>
#include "widgets/PhotoDirectoryWidget.h"
#include "widgets/CameraParametersWidget.h"
#include "widgets/IndicatorSpheresWidget.h"
#include "widgets/ObjectParametersWidget.h"
#include "widgets/RegionsOfInterestWidget.h"
#include "widgets/ReviewAndSelectWidget.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // Increase Qt image allocation limit (default is 128 MB)
    // Set to 512 MB to handle large images
    QImageReader::setAllocationLimit(512);

    setWindowTitle("cppPS");
    resize(800, 600);
    setupMenuBar();
    setupUi();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupMenuBar() {
    // Get the menu bar (QMainWindow automatically creates one)
    QMenuBar *menuBar = this->menuBar();

    // Create "Project" menu
    QMenu *projectMenu = menuBar->addMenu("Project");

    // Add menu items
    QAction *newAction = projectMenu->addAction("New Project");
    QAction *openAction = projectMenu->addAction("Open Project");
    QAction *saveAction = projectMenu->addAction("Save Project");

    // Connect actions to functions
    QObject::connect(newAction, &QAction::triggered,
                    [this]() { onNewProject(); });
    QObject::connect(openAction, &QAction::triggered,
                    [this]() { onOpenProject(); });
    QObject::connect(saveAction, &QAction::triggered,
                    [this]() { onSaveProject(); });
}

void MainWindow::onNewProject() {
    // Step 1: Get project name
    bool ok;
    QString projectName = QInputDialog::getText(this, "New Project",
                                               "Enter project name:",
                                               QLineEdit::Normal, "", &ok);
    if (!ok || projectName.isEmpty()) {
        return; // User cancelled or entered empty name
    }

    // Step 2: Select directory where project will be created
    QString documentsPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    QString parentDir = QFileDialog::getExistingDirectory(this,
                                                         "Select location for new project",
                                                         documentsPath);
    if (parentDir.isEmpty()) {
        return; // User cancelled
    }

    // Step 3: Create full project path
    QString projectPath = QDir(parentDir).filePath(projectName);

    // Check if directory already exists
    if (QDir(projectPath).exists()) {
        QMessageBox::warning(this, "Directory Exists",
                           "A directory with this name already exists. Please choose a different name.");
        return;
    }

    // Step 4: Create project directory structure
    QDir dir;
    if (!dir.mkpath(projectPath)) {
        QMessageBox::critical(this, "Error", "Failed to create project directory.");
        return;
    }

    // Create subdirectories
    QString photosDir = QDir(projectPath).filePath("photos");
    if (!dir.mkpath(photosDir)) {
        QMessageBox::critical(this, "Error", "Failed to create photos directory.");
        return;
    }

    // Step 5: Create settings JSON file
    QJsonObject settings;
    settings["projectName"] = projectName;
    settings["projectPath"] = projectPath;
    settings["photosDirectory"] = photosDir;
    settings["createdDate"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    settings["lastModified"] = QDateTime::currentDateTime().toString(Qt::ISODate);

    // Save settings to file
    QString settingsFilePath = QDir(projectPath).filePath("project_settings.json");
    QJsonDocument doc(settings);

    QFile settingsFile(settingsFilePath);
    if (!settingsFile.open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, "Error", "Failed to create settings file.");
        return;
    }

    settingsFile.write(doc.toJson());
    settingsFile.close();

    // Step 6: Success message
    QMessageBox::information(this, "Project Created",
                           QString("Project '%1' created successfully!\n\nLocation: %2")
                           .arg(projectName, projectPath));

    // Update window title to show current project
    setWindowTitle(QString("cppPS - %1").arg(projectName));

    // Update state variables
    m_currentProjectName = projectName;
    m_currentProjectPath = projectPath;

    m_photoDirectoryWidget->setPhotoDirectory(photosDir);
}

void MainWindow::onOpenProject() {
    QMessageBox::information(this, "Info", "Open Project clicked!");
}

void MainWindow::onSaveProject() {
    saveProjectConfig();
    QMessageBox::information(this, "Info", "Project settings saved!");
}


void MainWindow::setupUi()
{
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *mainVLayout = new QVBoxLayout(centralWidget); // Overall vertical layout

    // --- Top Area: Sidebar + Stacked Widget ---
    QHBoxLayout *contentHLayout = new QHBoxLayout();
    mainVLayout->addLayout(contentHLayout);

    // Left Sidebar (Categories)
    QListWidget* sidebarList = new QListWidget(this);
    sidebarList->addItem("Photo Directory");
    sidebarList->addItem("Camera Parameters");
    sidebarList->addItem("Indicator Spheres");
    sidebarList->addItem("Initial Guess");
    sidebarList->addItem("Regions of Interest");
    sidebarList->addItem("Review and Select ROI");
    sidebarList->setFixedWidth(180);
    contentHLayout->addWidget(sidebarList);

    // Right Main View (QStackedWidget with setting pages)
    stackedWidget = new QStackedWidget(this);

    // Create different content widgets for each category
    createContentWidgets();

    contentHLayout->addWidget(stackedWidget);

    // Connect sidebar selection to content change
    connect(sidebarList, &QListWidget::currentRowChanged,
            stackedWidget, &QStackedWidget::setCurrentIndex);

    // Set initial selection
    sidebarList->setCurrentRow(0);
}

void MainWindow::createContentWidgets()
{
    // Photo Directory page
    m_photoDirectoryWidget = new PhotoDirectoryWidget();
    m_photoDirectoryWidget->setPhotoDirectory("/path/to/your/photos");

    // Connect signals
    connect(m_photoDirectoryWidget, &PhotoDirectoryWidget::photoDirectoryChanged,
    this,[this](const QString& newPath) {
        m_problemConfig.photoDirectory = newPath;
        loadProjectConfig(); // Load config when directory changes
        saveProjectConfig(); // Auto-save after loading (creates .project dir if needed)
    });

    stackedWidget->addWidget(m_photoDirectoryWidget);


    // Camera Parameters page
    m_cameraParametersWidget = new CameraParametersWidget();

    // Connect camera parameters to config
    connect(m_cameraParametersWidget, &CameraParametersWidget::cameraConfigChanged,
    this, [this](const CameraConfig& config) {
        m_problemConfig.camera = config;
        saveProjectConfig(); // Auto-save when camera config changes
    });

    // Connect photo directory changes to camera parameters update
    connect(m_photoDirectoryWidget, &PhotoDirectoryWidget::photoDirectoryChanged,
            m_cameraParametersWidget, &CameraParametersWidget::updateFromDirectory);

    stackedWidget->addWidget(m_cameraParametersWidget);

    // Indicator Spheres page
    m_indicatorSpheresWidget = new IndicatorSpheresWidget();

    // Connect indicator spheres to config
    connect(m_indicatorSpheresWidget, &IndicatorSpheresWidget::indicatorConfigChanged,
    this, [this](const IndicatorSphereConfig& config) {
        m_problemConfig.indicators = config;
        saveProjectConfig(); // Auto-save when indicator config changes
    });

    // Connect photo directory changes to indicator spheres
    connect(m_photoDirectoryWidget, &PhotoDirectoryWidget::photoDirectoryChanged,
            m_indicatorSpheresWidget, &IndicatorSpheresWidget::setPhotoDirectory);

    stackedWidget->addWidget(m_indicatorSpheresWidget);

    // Initial Guess page (Object Parameters)
    m_objectParametersWidget = new ObjectParametersWidget();

    // Connect object parameters to config
    connect(m_objectParametersWidget, &ObjectParametersWidget::objectConfigChanged,
    this, [this](const ObjectConfig& config) {
        m_problemConfig.object = config;
        saveProjectConfig(); // Auto-save when object config changes
    });

    stackedWidget->addWidget(m_objectParametersWidget);

    // Regions of Interest page
    m_regionsOfInterestWidget = new RegionsOfInterestWidget();

    // Connect ROI widget to config
    connect(m_regionsOfInterestWidget, &RegionsOfInterestWidget::roisChanged,
    this, [this](const QVector<ROIConfig>& rois) {
        m_problemConfig.rois = rois;
        m_reviewAndSelectWidget->setRois(rois);
        saveProjectConfig(); // Auto-save when ROIs change
    });

    connect(m_regionsOfInterestWidget, &RegionsOfInterestWidget::selectedRoiChanged,
    this, [this](int index) {
        m_problemConfig.selectedRoiIndex = index;
        m_reviewAndSelectWidget->setSelectedRoiIndex(index);
        saveProjectConfig(); // Auto-save when selection changes
    });

    // Connect photo directory changes to ROI widget
    connect(m_photoDirectoryWidget, &PhotoDirectoryWidget::photoDirectoryChanged,
            m_regionsOfInterestWidget, &RegionsOfInterestWidget::setPhotoDirectory);

    stackedWidget->addWidget(m_regionsOfInterestWidget);

    // Review and Select page
    m_reviewAndSelectWidget = new ReviewAndSelectWidget();

    // Keep review page in sync with ROI editor
    connect(m_regionsOfInterestWidget, &RegionsOfInterestWidget::roisChanged,
            m_reviewAndSelectWidget, &ReviewAndSelectWidget::setRois);

    // Connect camera config to review widget
    connect(m_cameraParametersWidget, &CameraParametersWidget::cameraConfigChanged,
        m_reviewAndSelectWidget, &ReviewAndSelectWidget::setCameraProperties);

    // Persist selection changes made from review page
    connect(m_reviewAndSelectWidget, &ReviewAndSelectWidget::selectedRoiChanged,
            this, [this]() {
                m_problemConfig.selectedRoiIndex = m_reviewAndSelectWidget->selectedRoiIndex();
                saveProjectConfig();
            });

    stackedWidget->addWidget(m_reviewAndSelectWidget);
}

QString MainWindow::getProjectConfigPath() const {
    if (m_problemConfig.photoDirectory.isEmpty()) {
        return QString();
    }

    QDir photoDir(m_problemConfig.photoDirectory);
    QString projectDir = photoDir.filePath(".project");
    return QDir(projectDir).filePath("config.json");
}

void MainWindow::saveProjectConfig() {
    QString configPath = getProjectConfigPath();
    if (configPath.isEmpty()) {
        return;
    }

    // Create .project directory if it doesn't exist
    QFileInfo fileInfo(configPath);
    QDir dir = fileInfo.dir();
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    // Convert config to JSON
    QJsonDocument doc(m_problemConfig.toJson());

    // Write to file
    QFile file(configPath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson(QJsonDocument::Indented));
        file.close();
    }
}

void MainWindow::loadProjectConfig() {
    QString configPath = getProjectConfigPath();
    if (configPath.isEmpty()) {
        return;
    }

    QFile file(configPath);
    if (!file.exists() || !file.open(QIODevice::ReadOnly)) {
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (!doc.isNull() && doc.isObject()) {
        m_problemConfig = ProblemConfig::fromJson(doc.object());
        syncWidgetsFromConfig();
    }
}

void MainWindow::syncConfigFromWidgets() {
    // This is already being done through signal connections
}

void MainWindow::syncWidgetsFromConfig() {
    m_cameraParametersWidget->setConfig(m_problemConfig.camera);
    m_indicatorSpheresWidget->setConfig(m_problemConfig.indicators);
    m_objectParametersWidget->setConfig(m_problemConfig.object);
    m_regionsOfInterestWidget->setRois(m_problemConfig.rois);
    m_reviewAndSelectWidget->setRois(m_problemConfig.rois);
    m_reviewAndSelectWidget->setSelectedRoiIndex(m_problemConfig.selectedRoiIndex);
}
