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
#include "widgets/PhotoDirectoryWidget.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
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
    QMessageBox::information(this, "Info", "Save Project clicked!");
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
    QVBoxLayout* photoLayout = new QVBoxLayout(m_photoDirectoryWidget);
    photoLayout->addWidget(new QLabel("Photo Directory Settings"));
    // Add your photo directory controls here
    stackedWidget->addWidget(m_photoDirectoryWidget);

    // Camera Parameters page
    QWidget* cameraParamsWidget = new QWidget();
    QVBoxLayout* cameraLayout = new QVBoxLayout(cameraParamsWidget);
    cameraLayout->addWidget(new QLabel("Camera Parameters Settings"));
    // Add your camera parameter controls here
    stackedWidget->addWidget(cameraParamsWidget);

    // Indicator Spheres page
    QWidget* indicatorSpheresWidget = new QWidget();
    QVBoxLayout* spheresLayout = new QVBoxLayout(indicatorSpheresWidget);
    spheresLayout->addWidget(new QLabel("Indicator Spheres Settings"));
    // Add your indicator spheres controls here
    stackedWidget->addWidget(indicatorSpheresWidget);

    // Initial Guess page
    QWidget* initialGuessWidget = new QWidget();
    QVBoxLayout* guessLayout = new QVBoxLayout(initialGuessWidget);
    guessLayout->addWidget(new QLabel("Initial Guess Settings"));
    // Add your initial guess controls here
    stackedWidget->addWidget(initialGuessWidget);

    // Regions of Interest page
    QWidget* roiWidget = new QWidget();
    QVBoxLayout* roiLayout = new QVBoxLayout(roiWidget);
    roiLayout->addWidget(new QLabel("Regions of Interest Settings"));
    // Add your ROI controls here
    stackedWidget->addWidget(roiWidget);
}
